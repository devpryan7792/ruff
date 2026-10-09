//! Structural recursive types and their binding and substitution operations.
//!
//! A structural recursive type describes recursion within the type itself.
//! We write `μa. B` for a recursive type with body `B`, where occurrences of the recursive
//! variable `a` refer to the whole type. The `μa` is called a binder: it determines
//! what `a` refers to within `B`. For example:
//!
//! ```text
//! R = μa. tuple[int, a | None]
//! ```
//!
//! Such a recursive type is constructed, for example,
//! from the implicit type alias `R = tuple[int, "R | None"]`.
//!
//! A type is *closed* if every recursive variable is bound within that type;
//! otherwise, it is *open* (there is a dangling reference).
//! All types exposed to ordinary type operations must be closed, including during
//! intermediate normalization steps. The binding and substitution operations
//! in this module must maintain that invariant: an open body cannot be passed
//! directly to operations such as assignability checking.
//!
//! Substitution replaces occurrences of a variable with a type. It is
//! *capture-avoiding* when it preserves which binder every remaining variable refers
//! to, including variables in the inserted expression. We write `B[a := R]` for this
//! substitution of `R` for `a` in `B`. Unfolding exposes one layer of a recursive type
//! by substituting the whole type for its bound references:
//!
//! ```text
//! unfold(μa. B) = B[a := μa. B]
//! unfold(R) = tuple[int, R | None]
//! ```
//!
//! The result is closed, so ordinary type operations can inspect it. Binding works in
//! the other direction: it replaces applications of a recursive type with variables
//! and encloses the resulting open body in a `RecursiveType`. Both operations respect
//! nested binders. A variable refers to the nearest enclosing binder with the same
//! `RecursiveCycle`; substitution does not enter that binder's body when it binds the
//! target variable. The binder's arguments are outside its scope and are still substituted.
//!
//! A *type constructor* takes type arguments and produces a type. Recursive types can
//! also be parameterized this way. With type variable `T`, the alias
//! `Tree = tuple[T, "Tree[list[T]] | None"]` has the recursive type constructor:
//!
//! ```text
//! μF. λT. tuple[T, F[list[T]] | None]
//! ```
//!
//! Here `μF` binds the recursive constructor, and `λT` binds its type parameter.
//! The occurrence `F[list[T]]` is stored as `RecursiveVar` with the constructor's
//! `RecursiveCycle` and unspecialized arguments `[list[T]]`.
//!
//! To infer `x[1]` for `x: Tree[int]`, first unfold `x`'s type: substitute the constructor
//! for `F`, then apply `T := int`. This order closes the body before specializing it:
//!
//! ```text
//! unfold((μF. λT. tuple[T, F[list[T]] | None])[int])
//! = tuple[int, (μF. λT. tuple[T, F[list[T]] | None])[list[int]] | None]
//! = tuple[int, Tree[list[int]] | None]
//! ```
//!
//! Tuple subscripting then selects the element at index 1: `x[1]: Tree[list[int]] | None`.
//!
//! Cyclic value inference also needs references whose identities survive changes to their bodies.
//! For example, a decorator can return a tuple containing the decorated function. If each read
//! copies the previous tuple, every inference iteration can add another tuple layer. An inferred
//! recursive application identifies one output of its producing query; that query stores the
//! current body separately. Downstream substitutions are retained on the application and replayed
//! when an operation observes the body.
//!
//! Inference recovery preserves an outer constructor and widens references nested inside it.
//! References at the same structural depth can form mutually recursive equations, so their
//! normalization is a separate query with the original inference marker as its cycle value.

use std::cell::{Cell, RefCell};

use rustc_hash::FxHashSet;
use salsa::plumbing::AsId;
use ty_python_core::definition::Definition;
use ty_python_core::place::ScopedPlaceId;
use ty_python_core::place_table;
use ty_python_core::scope::ScopeId;
use ty_python_core::semantic_index;

use super::constraints::{ConstraintSet, IteratorConstraintsExtension};
use super::generics::{ApplySpecialization, Specialization};
use super::infer::DefinitionInferenceSource;
use super::relation::{TypeRelation, TypeRelationChecker};
use super::set_theoretic::TypeNormalization;
use super::type_alias::AliasCycleSummary;
use super::variance::{VarianceInferable, VarianceOrigin};
use super::visitor::{self, TypeVisitor};
use super::{
    ApplyTypeMappingVisitor, BindingContext, BoundTypeVarIdentity, BoundTypeVarInstance, ClassType,
    GenericContext, MaterializationKind, ProtocolInstanceType, SelfBinding, Type, TypeContext,
    TypeMapping, TypedDictType, VarianceTerm,
};
use crate::{Db, FxOrderSet, Program, ProgramEnvironment};

#[derive(Debug, Clone, PartialEq, Eq, Hash, get_size2::GetSize, salsa::SalsaValue)]
enum RecursiveSpecializationBase<'db> {
    Specialization {
        specialization: Specialization<'db>,
        specialize_self_domain: bool,
    },
    TypeAlias(Specialization<'db>),
    Partial {
        generic_context: GenericContext<'db>,
        types: Box<[Type<'db>]>,
        skip: Option<usize>,
    },
    Single(BoundTypeVarInstance<'db>, Type<'db>),
    ReturnCallables(Box<[(BoundTypeVarInstance<'db>, BoundTypeVarInstance<'db>)]>),
}

/// An owned substitution captured by a delayed operation. The keys retain their original
/// binding scopes; replacing a free variable after materialization must not move that
/// replacement beneath the materialization.
#[derive(Debug, Clone, PartialEq, Eq, Hash, get_size2::GetSize, salsa::SalsaValue)]
pub struct RecursiveSpecialization<'db> {
    base: RecursiveSpecializationBase<'db>,
    overrides: Box<[(BoundTypeVarInstance<'db>, Type<'db>)]>,
}

impl<'db> RecursiveSpecialization<'db> {
    fn capture(specialization: ApplySpecialization<'_, 'db>) -> Self {
        let base = match specialization {
            ApplySpecialization::Specialization {
                specialization,
                specialize_self_domain,
            } => RecursiveSpecializationBase::Specialization {
                specialization,
                specialize_self_domain,
            },
            ApplySpecialization::TypeAlias(specialization) => {
                RecursiveSpecializationBase::TypeAlias(specialization)
            }
            ApplySpecialization::Partial {
                generic_context,
                types,
                skip,
            } => RecursiveSpecializationBase::Partial {
                generic_context,
                types: types.into(),
                skip,
            },
            ApplySpecialization::Single(variable, ty) => {
                RecursiveSpecializationBase::Single(variable, ty)
            }
            ApplySpecialization::ReturnCallables(bindings) => {
                RecursiveSpecializationBase::ReturnCallables(
                    bindings
                        .iter()
                        .map(|(&variable, &ty)| (variable, ty))
                        .collect(),
                )
            }
            ApplySpecialization::WithBindings {
                specialization,
                bindings,
            } => {
                let Self { base, overrides } = Self::capture(*specialization);
                return Self {
                    base,
                    overrides: bindings.iter().chain(overrides.iter()).copied().collect(),
                };
            }
        };
        Self {
            base,
            overrides: Box::new([]),
        }
    }

    fn with_mapping<T>(&self, f: impl FnOnce(ApplySpecialization<'_, 'db>) -> T) -> T {
        let apply = |specialization| {
            if self.overrides.is_empty() {
                f(specialization)
            } else {
                f(ApplySpecialization::WithBindings {
                    specialization: &specialization,
                    bindings: &self.overrides,
                })
            }
        };
        match &self.base {
            RecursiveSpecializationBase::Specialization {
                specialization,
                specialize_self_domain,
            } => apply(ApplySpecialization::Specialization {
                specialization: *specialization,
                specialize_self_domain: *specialize_self_domain,
            }),
            RecursiveSpecializationBase::TypeAlias(specialization) => {
                apply(ApplySpecialization::TypeAlias(*specialization))
            }
            RecursiveSpecializationBase::Partial {
                generic_context,
                types,
                skip,
            } => apply(ApplySpecialization::Partial {
                generic_context: *generic_context,
                types,
                skip: *skip,
            }),
            RecursiveSpecializationBase::Single(variable, ty) => {
                apply(ApplySpecialization::Single(*variable, *ty))
            }
            RecursiveSpecializationBase::ReturnCallables(bindings) => {
                let bindings = bindings.iter().copied().collect();
                apply(ApplySpecialization::ReturnCallables(&bindings))
            }
        }
    }

    fn visit_types(&self, db: &'db dyn Db, visitor: &impl TypeVisitor<'db>) {
        match &self.base {
            RecursiveSpecializationBase::Specialization { specialization, .. }
            | RecursiveSpecializationBase::TypeAlias(specialization) => {
                super::generics::walk_specialization_types(db, *specialization, visitor);
            }
            RecursiveSpecializationBase::Partial { types, .. } => {
                for &ty in types {
                    visitor.visit_type(db, ty);
                }
            }
            RecursiveSpecializationBase::Single(_, ty) => visitor.visit_type(db, *ty),
            RecursiveSpecializationBase::ReturnCallables(bindings) => {
                for (_, ty) in bindings {
                    visitor.visit_type(db, Type::TypeVar(*ty));
                }
            }
        }
        for (_, ty) in &self.overrides {
            visitor.visit_type(db, *ty);
        }
    }

    fn map_types(
        &self,
        db: &'db dyn Db,
        mapping: &TypeMapping<'_, 'db>,
        visitor: &ApplyTypeMappingVisitor<'_, 'db>,
    ) -> Self {
        let map = |ty: Type<'db>| {
            ty.apply_type_mapping_impl(db, mapping, TypeContext::default(), visitor)
        };
        let base = match &self.base {
            RecursiveSpecializationBase::Specialization {
                specialization,
                specialize_self_domain,
            } => RecursiveSpecializationBase::Specialization {
                specialization: specialization.apply_type_mapping_impl(db, mapping, &[], visitor),
                specialize_self_domain: *specialize_self_domain,
            },
            RecursiveSpecializationBase::TypeAlias(specialization) => {
                RecursiveSpecializationBase::TypeAlias(specialization.apply_type_mapping_impl(
                    db,
                    mapping,
                    &[],
                    visitor,
                ))
            }
            RecursiveSpecializationBase::Partial {
                generic_context,
                types,
                skip,
            } => RecursiveSpecializationBase::Partial {
                generic_context: *generic_context,
                types: types.iter().copied().map(map).collect(),
                skip: *skip,
            },
            RecursiveSpecializationBase::Single(variable, ty) => {
                RecursiveSpecializationBase::Single(*variable, map(*ty))
            }
            RecursiveSpecializationBase::ReturnCallables(_) => self.base.clone(),
        };
        Self {
            base,
            overrides: self
                .overrides
                .iter()
                .map(|(variable, ty)| (*variable, map(*ty)))
                .collect(),
        }
    }
}

/// Operations on a recursive application, in evaluation order. Substitution remains
/// after an earlier materialization: `Top[tuple[T, Any]][T := Any]` exposes
/// `tuple[Any, object]`, while materializing after substitution exposes `tuple[object, object]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, get_size2::GetSize, salsa::SalsaValue)]
pub enum RecursiveOperation<'db> {
    Materialize(MaterializationKind, bool),
    Promote(super::PromotionMode, super::PromotionKind),
    Specialize(RecursiveSpecialization<'db>, Option<MaterializationKind>),
    BindLegacy(BindingContext<'db>),
    Freshen(GenericContext<'db>, u32),
    BindSelf(
        Type<'db>,
        Option<super::ClassLiteral<'db>>,
        Option<BindingContext<'db>>,
    ),
    ReplaceSelf(Type<'db>),
}

impl<'db> RecursiveOperation<'db> {
    pub(super) fn substitution(mapping: &TypeMapping<'_, 'db>) -> Option<Self> {
        Some(match mapping {
            TypeMapping::ApplySpecialization(specialization) => {
                Self::Specialize(RecursiveSpecialization::capture(*specialization), None)
            }
            TypeMapping::ApplySpecializationWithMaterialization {
                specialization,
                materialization_kind,
            } => Self::Specialize(
                RecursiveSpecialization::capture(*specialization),
                Some(*materialization_kind),
            ),
            TypeMapping::BindLegacyTypevars(context) => Self::BindLegacy(*context),
            TypeMapping::FreshenBoundTypeVars {
                generic_context,
                delta,
            } => Self::Freshen(*generic_context, *delta),
            TypeMapping::BindSelf(binding) => {
                Self::BindSelf(binding.ty, binding.class_literal, binding.binding_context)
            }
            TypeMapping::ReplaceSelf { new_upper_bound } => Self::ReplaceSelf(*new_upper_bound),
            _ => return None,
        })
    }

    fn visit_types(&self, db: &'db dyn Db, visitor: &impl TypeVisitor<'db>) {
        match self {
            Self::Specialize(specialization, _) => specialization.visit_types(db, visitor),
            Self::BindSelf(ty, ..) | Self::ReplaceSelf(ty) => visitor.visit_type(db, *ty),
            _ => {}
        }
    }

    pub(super) fn with_mapping<T>(&self, f: impl FnOnce(TypeMapping<'_, 'db>) -> T) -> T {
        match self {
            Self::Materialize(kind, _) => f(TypeMapping::Materialize(*kind)),
            Self::Promote(mode, kind) => f(TypeMapping::Promote(*mode, *kind)),
            Self::Specialize(specialization, kind) => {
                specialization.with_mapping(|specialization| {
                    f(match kind {
                        Some(kind) => TypeMapping::ApplySpecializationWithMaterialization {
                            specialization,
                            materialization_kind: *kind,
                        },
                        None => TypeMapping::ApplySpecialization(specialization),
                    })
                })
            }
            Self::BindLegacy(context) => f(TypeMapping::BindLegacyTypevars(*context)),
            Self::Freshen(generic_context, delta) => f(TypeMapping::FreshenBoundTypeVars {
                generic_context: *generic_context,
                delta: *delta,
            }),
            Self::BindSelf(ty, class_literal, binding_context) => {
                f(TypeMapping::BindSelf(SelfBinding {
                    ty: *ty,
                    class_literal: *class_literal,
                    binding_context: *binding_context,
                }))
            }
            Self::ReplaceSelf(new_upper_bound) => f(TypeMapping::ReplaceSelf {
                new_upper_bound: *new_upper_bound,
            }),
        }
    }

    pub(super) fn map_types(
        &self,
        db: &'db dyn Db,
        mapping: &TypeMapping<'_, 'db>,
        visitor: &ApplyTypeMappingVisitor<'_, 'db>,
    ) -> Self {
        let map = |ty: Type<'db>| {
            ty.apply_type_mapping_impl(db, mapping, TypeContext::default(), visitor)
        };
        match self {
            Self::Specialize(specialization, kind) => {
                Self::Specialize(specialization.map_types(db, mapping, visitor), *kind)
            }
            Self::BindSelf(ty, class, context) => Self::BindSelf(map(*ty), *class, *context),
            Self::ReplaceSelf(ty) => Self::ReplaceSelf(map(*ty)),
            operation => operation.clone(),
        }
    }
}

/// A recursive variable named by its binder's query cycle.
/// An escaping reference has no type semantics; in particular, it is neither a
/// gradual type nor an assignability operand.
/// Only recursive-type binding and substitution operations may construct or operate on recursive variables.
#[salsa::interned(debug, constructor=new_internal, heap_size=ruff_memory_usage::heap_size)]
pub struct RecursiveVar<'db> {
    /// Refers to the nearest enclosing recursive binder with this cycle identity.
    #[returns(copy)]
    cycle: RecursiveCycle,
    /// The unspecialized arguments of this occurrence, in the alias definition's scope.
    /// For `Tree = tuple[T, "Tree[list[T]] | None"]`, these are `[list[T]]`.
    /// Unfolding substitutes the enclosing application's arguments for the type parameters.
    #[returns(copy)]
    pub(super) arguments: Option<Specialization<'db>>,
    #[returns(ref)]
    operations: Box<[RecursiveOperation<'db>]>,
}

impl get_size2::GetSize for RecursiveVar<'_> {}

impl<'db> RecursiveVar<'db> {
    /// Unfold references to the target cycle, retaining variables bound by other cycles.
    pub(super) fn apply_type_mapping_impl(
        self,
        db: &'db dyn Db,
        mapping: &TypeMapping<'_, 'db>,
        visitor: &ApplyTypeMappingVisitor<'_, 'db>,
    ) -> Type<'db> {
        let arguments = self
            .arguments(db)
            .map(|arguments| arguments.apply_type_mapping_impl(db, mapping, &[], visitor));
        let operations: Box<[_]> = self
            .operations(db)
            .iter()
            .map(|operation| operation.map_types(db, mapping, visitor))
            .collect();
        match mapping {
            TypeMapping::ApplyRecursiveSubstitution(RecursiveMapping(
                RecursiveSubstitution::Unfold(recursive),
            )) if self.cycle(db) == recursive.cycle(db) => Type::Recursive(
                recursive
                    .with_arguments(db, arguments)
                    .with_operations(db, operations),
            ),
            TypeMapping::ApplyRecursiveSubstitution(_) => Type::RecursiveVar(Self::new_internal(
                db,
                self.cycle(db),
                arguments,
                operations,
            )),
            _ => unreachable!("semantic operation on an unbound recursive variable"),
        }
    }
}

/// A structural substitution that only the recursive-type binder can construct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, get_size2::GetSize)]
pub struct RecursiveMapping<'db>(RecursiveSubstitution<'db>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, get_size2::GetSize)]
enum RecursiveSubstitution<'db> {
    /// Replace references to a binder with applications of its recursive constructor.
    /// In the module example, this replaces `F[list[T]]` with `Tree[list[T]]`.
    Unfold(RecursiveType<'db>),
    /// Replace applications of the target binder with variables to form an open body.
    /// In the module example, this replaces `Tree[list[T]]` with `F[list[T]]`.
    Bind(RecursiveCycle),
}

impl RecursiveSubstitution<'_> {
    fn cycle(self, db: &dyn Db) -> RecursiveCycle {
        match self {
            Self::Unfold(recursive) => recursive.cycle(db),
            Self::Bind(cycle) => cycle,
        }
    }
}

/// Identifies an alias query and names its recursive binder and variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct RecursiveCycle(salsa::Id);

impl get_size2::GetSize for RecursiveCycle {}

impl RecursiveCycle {
    pub(crate) fn new(id: salsa::Id) -> Self {
        Self(id)
    }
}

/// Identifies one output of an inference query independently of its provisional value.
#[salsa::interned(debug, heap_size=ruff_memory_usage::heap_size)]
struct DefinitionCycle<'db> {
    query: RecursiveCycle,
    definition: Definition<'db>,
    declaration: bool,
}

/// Identifies a recursive constructor independently of its current application.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum RecursiveDefinitionIdentity<'db> {
    Definition(Definition<'db>),
    Inferred(RecursiveCycle),
}

/// Source metadata for named aliases and anonymous inference equations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, get_size2::GetSize, salsa::SalsaValue)]
pub enum RecursiveOrigin<'db> {
    Alias(Definition<'db>),
    Inferred(InferredOrigin<'db>),
}

/// Free-variable metadata for an inference result, retained without its provisional body.
/// Bound variables are formal arguments; runtime `TypeVar` objects remain captures until
/// a binding operation turns them into bound variables.
#[salsa::interned(debug, heap_size=ruff_memory_usage::heap_size)]
pub struct InferredOrigin<'db> {
    #[returns(copy)]
    program: Program<'db>,
    #[returns(copy)]
    query: RecursiveCycle,
    #[returns(ref)]
    captures: Box<[Type<'db>]>,
}

impl get_size2::GetSize for InferredOrigin<'_> {}

/// A closed recursive body or a reference to a separately cached inference result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, get_size2::GetSize, salsa::SalsaValue)]
pub enum RecursiveBody<'db> {
    Bound(Type<'db>),
    Definition {
        source: DefinitionInferenceSource<'db>,
        definition: Definition<'db>,
        declaration: bool,
    },
    Place {
        scope: ScopeId<'db>,
        place: ScopedPlaceId,
        explicit_reexport: bool,
        end_of_scope: bool,
    },
}

impl<'db> RecursiveBody<'db> {
    fn value(self, db: &'db dyn Db) -> Type<'db> {
        match self {
            Self::Bound(ty) => ty,
            Self::Definition {
                source,
                definition,
                declaration,
            } => source.body(db, definition, declaration),
            Self::Place {
                scope,
                place,
                explicit_reexport,
                end_of_scope,
            } => crate::place::recursive_place_body(
                db,
                scope,
                place,
                explicit_reexport,
                end_of_scope,
            ),
        }
    }
}

/// An application of a structural recursive type constructor.
/// The private body remains unspecialized; `arguments` records this application's
/// substitution for the constructor's type parameters. Unfolding replaces recursive
/// references with closed types, then applies that substitution before exposing the
/// result to ordinary type operations.
/// Use the binding operations in this module to construct recursive types.
#[salsa::interned(debug, constructor=new_internal, heap_size=ruff_memory_usage::heap_size)]
pub struct RecursiveType<'db> {
    /// The alias definition or the program and query that produce an inferred value.
    #[returns(copy)]
    pub(super) origin: RecursiveOrigin<'db>,
    /// Names the binder and distinguishes provisional types of different alias queries.
    #[returns(copy)]
    pub(super) cycle: RecursiveCycle,
    #[returns(copy)]
    body: RecursiveBody<'db>,
    /// The actual arguments for this application, which may themselves contain type variables.
    /// They are applied when unfolding; the stored body remains unspecialized.
    #[returns(copy)]
    base_arguments: Option<Specialization<'db>>,
    /// Operations applied after closing and specializing the stored body.
    #[returns(ref)]
    operations: Box<[RecursiveOperation<'db>]>,
}

impl get_size2::GetSize for RecursiveType<'_> {}

#[salsa::tracked]
impl<'db> RecursiveType<'db> {
    /// Summarize the open constructor body without applying semantic substitutions.
    pub(super) fn cycle_summary(self, db: &'db dyn Db) -> &'db AliasCycleSummary<'db> {
        #[salsa::tracked(
            returns(ref),
            cycle_initial=|db, id, _, ()| AliasCycleSummary::from_type(db, Type::divergent_alias(id)),
            heap_size=ruff_memory_usage::heap_size
        )]
        fn cycle_summary_impl<'db>(
            db: &'db dyn Db,
            recursive: RecursiveType<'db>,
            (): (),
        ) -> AliasCycleSummary<'db> {
            let mut summary = AliasCycleSummary::from_type(db, recursive.body(db).value(db));
            // Nested bodies can refer to an enclosing binder. Close only the cycle marker,
            // so recovery never exposes an unbound variable as a standalone type.
            if let Some(Type::RecursiveVar(variable)) = summary.cycle {
                summary.cycle = Some(Type::divergent_alias(variable.cycle(db).0));
            }
            summary
        }

        cycle_summary_impl(db, self.constructor(db), ())
    }

    /// Seed a query cycle with `μa. a`, using the same identity for binder and variable.
    pub(super) fn initial(
        db: &'db dyn Db,
        definition: Definition<'db>,
        cycle: salsa::Id,
        parameters: Option<GenericContext<'db>>,
    ) -> Self {
        let cycle = RecursiveCycle(cycle);
        let arguments = parameters.map(|parameters| parameters.identity_specialization(db));
        Self::new_internal(
            db,
            RecursiveOrigin::Alias(definition),
            cycle,
            RecursiveBody::Bound(Type::RecursiveVar(RecursiveVar::new_internal(
                db,
                cycle,
                arguments,
                Box::<[RecursiveOperation<'_>]>::default(),
            ))),
            arguments,
            Box::<[RecursiveOperation<'_>]>::default(),
        )
    }

    /// Close recursive occurrences after inferring an alias's constructor expression.
    pub(super) fn recover(
        db: &'db dyn Db,
        definition: Definition<'db>,
        cycle: salsa::Id,
        parameters: Option<GenericContext<'db>>,
        result: Type<'db>,
    ) -> Type<'db> {
        // Shared dependencies can still contain older iterations of this alias. They refer
        // to the same definition even when their provisional bodies differ. Derive the binder
        // from the query inputs: an earlier result might not expose recursive references yet.
        Self::initial(db, definition, cycle, parameters).bind(
            db,
            &ProgramEnvironment::from_definition(definition),
            result,
        )
    }

    /// Bind references to this alias's query cycle in a closed inference result.
    /// Each bound occurrence retains its arguments and refers to this constructor's cycle.
    fn bind(
        self,
        db: &'db dyn Db,
        env: &ProgramEnvironment<'db>,
        original: Type<'db>,
    ) -> Type<'db> {
        let body = original.apply_type_mapping_impl(
            db,
            &TypeMapping::ApplyRecursiveSubstitution(RecursiveMapping(
                RecursiveSubstitution::Bind(self.cycle(db)),
            )),
            TypeContext::default(),
            &ApplyTypeMappingVisitor::new(env),
        );
        // Alias arguments can expose a reference without introducing a container.
        if body.has_unguarded_alias_cycle(db) {
            Type::divergent_alias(self.cycle(db).0)
        } else if body == original {
            // Binding changes a closed type only by introducing references to this binder.
            body
        } else {
            Type::Recursive(Self::new_internal(
                db,
                self.origin(db),
                self.cycle(db),
                RecursiveBody::Bound(body),
                self.arguments(db),
                Box::<[RecursiveOperation<'_>]>::default(),
            ))
        }
    }

    /// Preserve the query identity independently of its current approximation.
    pub(super) fn inferred_definition(
        db: &'db dyn Db,
        env: &ProgramEnvironment<'db>,
        cycle: salsa::Id,
        source: DefinitionInferenceSource<'db>,
        definition: Definition<'db>,
        declaration: bool,
        result: Type<'db>,
    ) -> Type<'db> {
        let binder = RecursiveCycle(
            DefinitionCycle::new(db, RecursiveCycle(cycle), definition, declaration).as_id(),
        );
        Self::inferred_reference(
            db,
            env,
            binder,
            RecursiveCycle(cycle),
            RecursiveBody::Definition {
                source,
                definition,
                declaration,
            },
            result,
        )
    }

    /// Preserve the identity of a public-place query whose result is still being inferred.
    pub(crate) fn inferred_place(
        db: &'db dyn Db,
        cycle: RecursiveCycle,
        scope: ScopeId<'db>,
        place: ScopedPlaceId,
        explicit_reexport: bool,
        end_of_scope: bool,
        result: Type<'db>,
    ) -> Type<'db> {
        let env = ProgramEnvironment::from_scope(scope);
        Self::inferred_reference(
            db,
            &env,
            cycle,
            cycle,
            RecursiveBody::Place {
                scope,
                place,
                explicit_reexport,
                end_of_scope,
            },
            result,
        )
    }

    /// Represent structural results without embedding their provisional bodies in the key.
    /// Leaves keep their existing identities. The application's parameters include lexical
    /// variables and variables in the current result so substitutions do not change the query
    /// that constructs the body.
    fn inferred_reference(
        db: &'db dyn Db,
        env: &ProgramEnvironment<'db>,
        binder: RecursiveCycle,
        query: RecursiveCycle,
        body: RecursiveBody<'db>,
        result: Type<'db>,
    ) -> Type<'db> {
        struct Children<'a, 'db> {
            env: &'a ProgramEnvironment<'db>,
            found: Cell<bool>,
        }
        impl<'db> TypeVisitor<'db> for Children<'_, 'db> {
            fn program_environment(&self) -> &ProgramEnvironment<'db> {
                self.env
            }
            fn should_visit_lazy_type_attributes(&self) -> bool {
                false
            }
            fn notify_skipped_lazy_type_attributes(&self) {
                self.found.set(true);
            }
            fn visit_type(&self, _: &'db dyn Db, _: Type<'db>) {
                self.found.set(true);
            }
        }
        let visitor::TypeKind::NonAtomic(structure) = result.into() else {
            return result;
        };
        let children = Children {
            env,
            found: Cell::new(false),
        };
        visitor::walk_non_atomic_type(db, structure, &children);
        if !children.found.get() {
            return result;
        }
        let mut variables = FxOrderSet::default();
        let scope = match body {
            RecursiveBody::Definition { source, .. } => Some(source.scope(db)),
            RecursiveBody::Place { scope, .. } => Some(scope),
            RecursiveBody::Bound(_) => None,
        };
        if let Some(scope) = scope {
            let index = semantic_index(db, scope.program_file(db));
            for (_, scope) in index.ancestor_scopes(scope.file_scope_id(db)) {
                if let Some(context) = GenericContext::lexical_of_node(db, scope.node(), index) {
                    variables.extend(context.variables(db));
                }
            }
        }
        let (stored, _) = stored_variables(db, env, result);
        let mut captures = Vec::new();
        for variable in stored {
            if let Type::TypeVar(variable) = variable {
                variables.insert(variable);
            } else {
                captures.push(variable);
            }
        }
        let arguments = (!variables.is_empty()).then(|| {
            GenericContext::from_typevar_instances(db, env, variables).identity_specialization(db)
        });
        Type::Recursive(Self::new_internal(
            db,
            RecursiveOrigin::Inferred(InferredOrigin::new(
                db,
                env.program(db),
                query,
                captures.into_boxed_slice(),
            )),
            binder,
            body,
            arguments,
            Box::<[RecursiveOperation<'_>]>::default(),
        ))
    }

    pub(super) fn with_arguments(
        self,
        db: &'db dyn Db,
        arguments: Option<Specialization<'db>>,
    ) -> Self {
        Self::new_internal(
            db,
            self.origin(db),
            self.cycle(db),
            self.body(db),
            arguments,
            self.operations(db).clone(),
        )
    }

    fn with_operations(self, db: &'db dyn Db, operations: Box<[RecursiveOperation<'db>]>) -> Self {
        Self::new_internal(
            db,
            self.origin(db),
            self.cycle(db),
            self.body(db),
            self.base_arguments(db),
            operations,
        )
    }

    /// Project the current arguments for naming and parameter-flow analysis. Materialization
    /// belongs to the resulting type and does not rewrite its nominal arguments.
    #[salsa::tracked(returns(copy), heap_size=ruff_memory_usage::heap_size)]
    pub(super) fn arguments(self, db: &'db dyn Db) -> Option<Specialization<'db>> {
        let mut arguments = self.base_arguments(db)?;
        let env = self.environment(db);
        for operation in self.operations(db) {
            if matches!(operation, RecursiveOperation::Materialize(..)) {
                continue;
            }
            operation.with_mapping(|mapping| {
                arguments = arguments.apply_type_mapping_impl(
                    db,
                    &mapping,
                    &[],
                    &ApplyTypeMappingVisitor::new_for_type_construction(&env),
                );
            });
        }
        Some(arguments)
    }

    /// The outermost materialization, when no later substitution can introduce gradual types.
    pub(super) fn materialization_kind(self, db: &'db dyn Db) -> Option<MaterializationKind> {
        match self.operations(db).last() {
            Some(RecursiveOperation::Materialize(kind, _)) => Some(*kind),
            _ => None,
        }
    }

    /// Whether argument-based comparisons need the complete operation sequence.
    /// Ordered substitutions and materializations that exclude variable metadata must be
    /// observed by replaying their operations on the closed body.
    pub(super) fn requires_operation_replay(self, db: &'db dyn Db) -> bool {
        !matches!(
            self.operations(db).as_ref(),
            [] | [RecursiveOperation::Materialize(_, true)]
        )
    }

    /// Whether a substitution changes a variable captured from outside this constructor.
    /// Formal parameters are substituted through the application's arguments instead.
    fn captures_change(
        self,
        db: &'db dyn Db,
        mapping: &TypeMapping<'_, 'db>,
        visitor: &ApplyTypeMappingVisitor<'_, 'db>,
    ) -> bool {
        let (variables, complete) = self.constructor(db).captured_variables(db);
        if !*complete {
            return true;
        }
        variables.iter().copied().any(|mut variable| {
            for operation in self.operations(db) {
                operation.with_mapping(|mapping| {
                    let mut operation_visitor =
                        ApplyTypeMappingVisitor::new_for_type_construction(visitor.env);
                    if let RecursiveOperation::Materialize(_, map_bounds) = operation {
                        operation_visitor.materialize_typevar_bounds_and_defaults = *map_bounds;
                    }
                    variable = variable.apply_type_mapping_impl(
                        db,
                        &mapping,
                        TypeContext::default(),
                        &operation_visitor,
                    );
                });
            }
            variable.apply_type_mapping_impl(db, mapping, TypeContext::default(), visitor)
                != variable
        })
    }

    #[salsa::tracked(
        returns(ref),
        cycle_initial=|_, _, _| (Box::default(), true),
        heap_size=ruff_memory_usage::heap_size
    )]
    fn captured_variables(self, db: &'db dyn Db) -> (Box<[Type<'db>]>, bool) {
        if let RecursiveOrigin::Inferred(origin) = self.origin(db) {
            // Observing the producing query here would make substitution depend on its
            // changing body and defeat the stable identity used for inference convergence.
            return (origin.captures(db).clone(), true);
        }
        let body = self.body(db).value(db);
        let (variables, complete) = stored_variables(db, &self.environment(db), body);
        let variables = variables
            .into_iter()
            .filter(|variable| {
                !matches!(variable, Type::TypeVar(bound) if self.parameters(db)
                .is_some_and(|parameters| parameters.contains(db, bound.identity(db))))
            })
            .collect();
        (variables, complete)
    }

    fn with_materialization(
        self,
        db: &'db dyn Db,
        materialization: Option<MaterializationKind>,
    ) -> Self {
        let mut operations = self.operations(db).to_vec();
        if matches!(
            operations.last(),
            Some(RecursiveOperation::Materialize(_, _))
        ) {
            operations.pop();
        }
        if let Some(kind) = materialization {
            operations.push(RecursiveOperation::Materialize(kind, true));
        }
        self.with_operations(db, operations.into_boxed_slice())
    }

    /// Parameters bound by this recursive type constructor.
    pub(super) fn parameters(self, db: &'db dyn Db) -> Option<GenericContext<'db>> {
        self.base_arguments(db)
            .map(|arguments| arguments.generic_context(db))
    }

    /// Identify the constructor before traversing its declaration or inference body.
    pub(super) fn definition_identity(self, db: &'db dyn Db) -> RecursiveDefinitionIdentity<'db> {
        match self.origin(db) {
            RecursiveOrigin::Alias(definition) => {
                RecursiveDefinitionIdentity::Definition(definition)
            }
            RecursiveOrigin::Inferred(..) => RecursiveDefinitionIdentity::Inferred(self.cycle(db)),
        }
    }

    pub(super) fn inferred(self, db: &'db dyn Db) -> bool {
        matches!(self.origin(db), RecursiveOrigin::Inferred(..))
    }

    /// Expose the outer structure during inference recovery. Normalizing an unguarded
    /// reference can return to this query, while nested references are widened by the caller.
    #[salsa::tracked(
        returns(copy),
        cycle_initial=|_, _, _, marker: Type<'db>| marker,
        heap_size=ruff_memory_usage::heap_size
    )]
    pub(super) fn normalize_inference_cycle(self, db: &'db dyn Db, marker: Type<'db>) -> Type<'db> {
        let env = self.environment(db);
        self.unfold(db, &env)
            .map(|body| {
                body.recursive_type_normalized_impl(db, &env, marker, false)
                    .unwrap_or(marker)
            })
            .unwrap_or(marker)
    }

    /// Recover the producing query's marker independently of this output's binder identity.
    pub(super) fn inference_cycle_marker(self, db: &'db dyn Db) -> Option<Type<'db>> {
        match self.origin(db) {
            RecursiveOrigin::Inferred(origin) => Some(Type::divergent(origin.query(db).0)),
            RecursiveOrigin::Alias(_) => None,
        }
    }

    /// The source alias's definition and name, if this binder comes from an alias.
    pub(super) fn alias(self, db: &'db dyn Db) -> Option<(Definition<'db>, &'db str)> {
        let RecursiveOrigin::Alias(definition) = self.origin(db) else {
            return None;
        };
        let name = place_table(db, definition.scope(db))
            .symbol(definition.place(db).expect_symbol())
            .name();
        Some((definition, name))
    }

    /// Restore the formal arguments and remove materialization for constructor analysis.
    pub(super) fn constructor(self, db: &'db dyn Db) -> Self {
        // Like an unspecialized PEP 695 alias, parameter-flow analysis must not
        // re-enter materialization while deriving the constructor's identity.
        self.with_operations(db, Box::<[RecursiveOperation<'_>]>::default())
            .with_arguments(
                db,
                self.parameters(db)
                    .map(|parameters| parameters.identity_specialization(db)),
            )
    }

    /// The program in which the recursive type's body was constructed.
    pub fn environment(self, db: &'db dyn Db) -> ProgramEnvironment<'db> {
        match self.origin(db) {
            RecursiveOrigin::Alias(definition) => ProgramEnvironment::from_definition(definition),
            RecursiveOrigin::Inferred(origin) => {
                ProgramEnvironment::from_program(origin.program(db))
            }
        }
    }

    /// Substitute closed types for references before exposing the body.
    ///
    /// Report whether unfolding returns exactly `Type::Recursive(self)`. An unfolded
    /// type can still contain recursive references, so callers must retain their recursion guards.
    pub fn unfold(self, db: &'db dyn Db, env: &ProgramEnvironment<'db>) -> UnfoldResult<'db> {
        let unfolded = self.replay_operations(db, env, self.unfolded_body(db));
        if unfolded == Type::Recursive(self) {
            UnfoldResult::Unchanged(self)
        } else {
            UnfoldResult::Unfolded(unfolded)
        }
    }

    /// The unspecialized finite body used to analyze tuple unpack dependencies.
    pub(super) fn shape_body(self, db: &'db dyn Db) -> Type<'db> {
        self.body(db).value(db)
    }

    /// Instantiate one body node with the same environment and operation order as unfolding.
    pub(super) fn apply_to_node_structural(self, db: &'db dyn Db, node: Type<'db>) -> Type<'db> {
        self.replay_operations(db, &self.environment(db), self.close_body_node(db, node))
    }

    fn replay_operations(
        self,
        db: &'db dyn Db,
        env: &ProgramEnvironment<'db>,
        mut unfolded: Type<'db>,
    ) -> Type<'db> {
        for operation in self.operations(db) {
            operation.with_mapping(|mapping| {
                let mut visitor = ApplyTypeMappingVisitor::new_for_type_construction(env);
                if let RecursiveOperation::Materialize(_, map_bounds) = operation {
                    visitor.materialize_typevar_bounds_and_defaults = *map_bounds;
                }
                unfolded = unfolded.apply_type_mapping_impl(
                    db,
                    &mapping,
                    TypeContext::default(),
                    &visitor,
                );
            });
        }
        unfolded
    }

    /// Share the closed, specialized body across mappings with different visitors.
    #[salsa::tracked(
        returns(copy),
        cycle_initial=|_, _, recursive: RecursiveType<'db>| Type::Recursive(recursive),
        heap_size=ruff_memory_usage::heap_size
    )]
    fn unfolded_body(self, db: &'db dyn Db) -> Type<'db> {
        self.close_body_node(db, self.body(db).value(db))
    }

    fn close_body_node(self, db: &'db dyn Db, node: Type<'db>) -> Type<'db> {
        let env = self.environment(db);
        let unfolded = node.apply_type_mapping_impl(
            db,
            &TypeMapping::ApplyRecursiveSubstitution(RecursiveMapping(
                RecursiveSubstitution::Unfold(
                    self.with_operations(db, Box::<[RecursiveOperation<'_>]>::default()),
                ),
            )),
            TypeContext::default(),
            &ApplyTypeMappingVisitor::new_for_type_construction(&env),
        );
        match self.base_arguments(db) {
            Some(arguments) => {
                let specialization = ApplySpecialization::TypeAlias(arguments);
                let mapping = match arguments.materialization_kind(db) {
                    Some(materialization_kind) => {
                        TypeMapping::ApplySpecializationWithMaterialization {
                            specialization,
                            materialization_kind,
                        }
                    }
                    None => TypeMapping::ApplySpecialization(specialization),
                };
                unfolded.apply_type_mapping_impl(
                    db,
                    &mapping,
                    TypeContext::default(),
                    &ApplyTypeMappingVisitor::new_for_type_construction(&env),
                )
            }
            None => unfolded,
        }
    }

    /// Substitute by cycle identity, respecting the scope of nested recursive binders.
    pub(super) fn apply_type_mapping_impl(
        self,
        db: &'db dyn Db,
        mapping: &TypeMapping<'_, 'db>,
        tcx: TypeContext<'db>,
        visitor: &ApplyTypeMappingVisitor<'_, 'db>,
    ) -> Type<'db> {
        if matches!(mapping, TypeMapping::Normalize) {
            let recursive = self.with_arguments(
                db,
                self.base_arguments(db)
                    .map(|arguments| arguments.apply_type_mapping_impl(db, mapping, &[], visitor)),
            );
            let operations = self
                .operations(db)
                .iter()
                .map(|operation| operation.map_types(db, mapping, visitor))
                .collect();
            return Type::Recursive(recursive.with_operations(db, operations));
        }
        match mapping {
            TypeMapping::ApplyRecursiveSubstitution(RecursiveMapping(
                RecursiveSubstitution::Bind(cycle),
            )) if self.cycle(db) == *cycle => {
                let arguments = self
                    .base_arguments(db)
                    .map(|arguments| arguments.apply_type_mapping_impl(db, mapping, &[], visitor));
                let operations: Box<[_]> = self
                    .operations(db)
                    .iter()
                    .map(|operation| operation.map_types(db, mapping, visitor))
                    .collect();
                Type::RecursiveVar(RecursiveVar::new_internal(
                    db,
                    self.cycle(db),
                    arguments,
                    operations,
                ))
            }
            TypeMapping::ApplyRecursiveSubstitution(RecursiveMapping(substitution)) => {
                // This binder shadows the target in its body, but not in its arguments.
                let body = match self.body(db) {
                    RecursiveBody::Bound(body) if self.cycle(db) != substitution.cycle(db) => {
                        RecursiveBody::Bound(
                            body.apply_type_mapping_impl(db, mapping, tcx, visitor),
                        )
                    }
                    body => body,
                };
                let arguments = self
                    .base_arguments(db)
                    .map(|arguments| arguments.apply_type_mapping_impl(db, mapping, &[], visitor));
                Type::Recursive(Self::new_internal(
                    db,
                    self.origin(db),
                    self.cycle(db),
                    body,
                    arguments,
                    self.operations(db)
                        .iter()
                        .map(|operation| operation.map_types(db, mapping, visitor))
                        .collect::<Box<[_]>>(),
                ))
            }
            TypeMapping::Promote(mode, kind) if self.inferred(db) => {
                let operation = RecursiveOperation::Promote(*mode, *kind);
                if self
                    .operations(db)
                    .iter()
                    .rev()
                    .take_while(|operation| matches!(operation, RecursiveOperation::Promote(..)))
                    .any(|previous| previous == &operation)
                {
                    return Type::Recursive(self);
                }
                let mut operations = self.operations(db).to_vec();
                operations.push(operation);
                Type::Recursive(self.with_operations(db, operations.into_boxed_slice()))
            }
            mapping if let Some(operation) = RecursiveOperation::substitution(mapping) => {
                // These mappings substitute free variables, which are captured by the alias's
                // arguments. Its formal body must remain independent of the calling context.
                let structural;
                let visitor = if visitor.normalization == TypeNormalization::Semantic {
                    structural = visitor.for_type_construction();
                    &structural
                } else {
                    visitor
                };
                let arguments = self
                    .arguments(db)
                    .map(|arguments| arguments.apply_type_mapping_impl(db, mapping, &[], visitor));
                let captures_change = self.captures_change(db, mapping, visitor);
                if self.operations(db).is_empty() && !captures_change {
                    return Type::Recursive(self.with_arguments(db, arguments));
                }
                if arguments == self.arguments(db) && !captures_change {
                    return Type::Recursive(self);
                }
                let mut operations = self.operations(db).to_vec();
                operations.push(operation);
                Type::Recursive(self.with_operations(db, operations.into_boxed_slice()))
            }
            TypeMapping::Materialize(_)
                if let Some(RecursiveOperation::Materialize(_, map_bounds)) =
                    self.operations(db).last()
                    && (*map_bounds || !visitor.materialize_typevar_bounds_and_defaults) =>
            {
                Type::Recursive(self)
            }
            TypeMapping::Materialize(kind) => {
                if structurally_static(db, visitor.env, Type::Recursive(self)) {
                    return Type::Recursive(self);
                }
                let mut operations = self.operations(db).to_vec();
                operations.push(RecursiveOperation::Materialize(
                    *kind,
                    visitor.materialize_typevar_bounds_and_defaults,
                ));
                Type::Recursive(self.with_operations(db, operations.into_boxed_slice()))
            }
            TypeMapping::EagerExpansion => {
                visitor.visit(db, Type::Recursive(self), mapping, || {
                    // Expand arguments only where the body exposes them. Expanding stored arguments
                    // first can feed a recursive alias's previous approximation into its own arguments.
                    self.unfold(db, visitor.env)
                        .map(|unfolded| {
                            let mapped =
                                unfolded.apply_type_mapping_impl(db, mapping, tcx, visitor);
                            if mapped == unfolded {
                                Type::Recursive(self)
                            } else {
                                mapped
                            }
                        })
                        .into_type()
                })
            }
            _ => visitor.visit(db, Type::Recursive(self), mapping, || {
                // Map arguments before unfolding so recursive backedges retain their mapped
                // arguments. Pending operations must run first: rewriting their input arguments
                // would move this mapping across a captured materialization.
                let recursive = if self.operations(db).is_empty() {
                    let arguments = self.arguments(db).map(|arguments| {
                        arguments.apply_type_mapping_impl(db, mapping, &[], visitor)
                    });
                    self.with_arguments(db, arguments)
                } else {
                    self
                };
                recursive
                    .unfold(db, visitor.env)
                    .map(|unfolded| {
                        let mapped = unfolded.apply_type_mapping_impl(db, mapping, tcx, visitor);
                        if mapped == unfolded {
                            Type::Recursive(recursive)
                        } else {
                            mapped
                        }
                    })
                    .into_type()
            }),
        }
    }

    pub(in crate::types) fn variance_equation(
        self,
        db: &'db dyn Db,
        typevar: BoundTypeVarIdentity<'db>,
    ) -> VarianceTerm<'db> {
        let env = self.environment(db);
        self.unfold(db, &env)
            .map(|unfolded| unfolded.variance_of(db, &env, typevar))
            .unwrap_or(VarianceTerm::BIVARIANT)
    }
}

/// The outcome of unfolding one layer of a [`RecursiveType`].
///
/// Mapping transforms the unfolded value while retaining the original recursive type
/// when unfolding made no progress.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub enum UnfoldResult<'db, T = Type<'db>> {
    /// The unfolded type, or a value produced by mapping it.
    Unfolded(T),
    /// Unfolding returns the original recursive type, for example for `μa. a`.
    Unchanged(RecursiveType<'db>),
}

impl<'db> UnfoldResult<'db, Type<'db>> {
    /// Return the unfolded or mapped type, or the original recursive type if unfolding made no progress.
    ///
    /// Callers that recursively process the returned type must use their own recursion guards.
    /// Unfolding one layer does not eliminate cycles, even when it makes progress.
    #[inline]
    pub fn into_type(self) -> Type<'db> {
        match self {
            Self::Unfolded(ty) => ty,
            Self::Unchanged(recursive) => Type::Recursive(recursive),
        }
    }
}

impl<'db, T> UnfoldResult<'db, T> {
    /// Return the contained value if unfolding made progress.
    #[inline]
    pub(crate) fn into_unfolded(self) -> Option<T> {
        match self {
            Self::Unfolded(value) => Some(value),
            Self::Unchanged(_) => None,
        }
    }

    /// Return whether unfolding made progress and the contained value satisfies `predicate`.
    #[inline]
    #[expect(
        clippy::wrong_self_convention,
        reason = "Like Option::is_some_and, the predicate consumes the contained value."
    )]
    pub(crate) fn is_unfolded_and(self, predicate: impl FnOnce(T) -> bool) -> bool {
        match self {
            Self::Unfolded(value) => predicate(value),
            Self::Unchanged(_) => false,
        }
    }

    /// Return whether unfolding made no progress or the contained value satisfies `predicate`.
    #[inline]
    #[expect(
        clippy::wrong_self_convention,
        reason = "Like Option::is_none_or, the predicate consumes the contained value."
    )]
    pub(crate) fn is_unchanged_or(self, predicate: impl FnOnce(T) -> bool) -> bool {
        match self {
            Self::Unfolded(value) => predicate(value),
            Self::Unchanged(_) => true,
        }
    }

    /// Transform the contained value, preserving the original recursive type if unfolding made no progress.
    #[inline]
    pub(crate) fn map<U>(self, operation: impl FnOnce(T) -> U) -> UnfoldResult<'db, U> {
        match self {
            Self::Unfolded(value) => UnfoldResult::Unfolded(operation(value)),
            Self::Unchanged(recursive) => UnfoldResult::Unchanged(recursive),
        }
    }

    /// Return the contained value, or call `fallback` if unfolding made no progress.
    #[inline]
    pub(crate) fn unwrap_or_else(self, fallback: impl FnOnce() -> T) -> T {
        match self {
            Self::Unfolded(value) => value,
            Self::Unchanged(_) => fallback(),
        }
    }

    /// Return the contained value, or return `fallback` if unfolding made no progress.
    #[inline]
    pub(crate) fn unwrap_or(self, fallback: T) -> T {
        match self {
            Self::Unfolded(value) => value,
            Self::Unchanged(_) => fallback,
        }
    }
}

impl<'c, 'db> TypeRelationChecker<'_, 'c, 'db> {
    /// Prove a relation between applications of the same recursive constructor
    /// (same unspecialized body) based on inclusion or overlap of their type
    /// arguments' materialization families.
    ///
    /// For example, every materialization of `R[int]` is also a materialization of
    /// `R[Any]`: choosing `int` restricts the possibilities for the argument without
    /// changing the constructor's body. Comparing arguments lets us establish
    /// relations between the applications' top and bottom materializations without
    /// unfolding their bodies, even when unfolding would keep growing the arguments.
    ///
    /// Argument comparison is sufficient but not necessary to establish the relation:
    /// different arguments can produce equivalent alias types. If this check cannot
    /// establish the relation, the caller can still unfold and compare the bodies.
    pub(super) fn when_recursive_types_relate_by_arguments(
        &self,
        db: &'db dyn Db,
        source: RecursiveType<'db>,
        target: RecursiveType<'db>,
    ) -> ConstraintSet<'db, 'c> {
        // Invariant comparisons preserve variable metadata. A materialization constructed
        // in that same mode still describes its arguments' materialization families;
        // substitutions or changes of mode require observing the complete operation sequence.
        let needs_body = |recursive: RecursiveType<'db>| {
            recursive.requires_operation_replay(db)
                && !matches!(
                    recursive.operations(db).as_ref(),
                    [RecursiveOperation::Materialize(_, false)]
                        if !self.materialization_visitor.materialize_typevar_bounds_and_defaults
                )
        };
        if needs_body(source)
            || needs_body(target)
            || !matches!(
                self.relation,
                TypeRelation::Subtyping | TypeRelation::Assignability
            )
            || source.constructor(db) != target.constructor(db)
        {
            return self.never();
        }
        let (Some(source_arguments), Some(target_arguments)) =
            (source.arguments(db), target.arguments(db))
        else {
            return self.never();
        };
        // This proof substitutes plain parameter values. Variadic packs and
        // materialized specializations carry additional substitution semantics.
        if source_arguments.generic_context(db) != target_arguments.generic_context(db)
            || source_arguments.materialization_kind(db).is_some()
            || target_arguments.materialization_kind(db).is_some()
            || source_arguments.tuple(db).is_some()
            || target_arguments.tuple(db).is_some()
        {
            return self.never();
        }
        let source_kind =
            source
                .materialization_kind(db)
                .unwrap_or(if self.relation.is_assignability() {
                    MaterializationKind::Bottom
                } else {
                    MaterializationKind::Top
                });
        let target_kind =
            target
                .materialization_kind(db)
                .unwrap_or(if self.relation.is_assignability() {
                    MaterializationKind::Top
                } else {
                    MaterializationKind::Bottom
                });
        // Top-to-bottom also requires a static body: equal arguments do not rule
        // out a fixed `Any`. Unchanged materializations retain the original binder.
        let unmaterialized_target = Type::Recursive(target.with_materialization(db, None));
        if matches!(
            (source_kind, target_kind),
            (MaterializationKind::Top, MaterializationKind::Bottom)
        ) && unmaterialized_target.materialize(
            db,
            MaterializationKind::Top,
            self.materialization_visitor,
        ) != unmaterialized_target.materialize(
            db,
            MaterializationKind::Bottom,
            self.materialization_visitor,
        ) {
            return self.never();
        }
        source_arguments
            .types(db)
            .iter()
            .zip(target_arguments.types(db))
            .when_all(db, self.constraints, |(source, target)| {
                self.check_subtyping_in_invariant_position(
                    db,
                    *source,
                    source_kind,
                    *target,
                    target_kind,
                )
            })
    }
}

impl<'db> VarianceInferable<'db> for RecursiveType<'db> {
    fn variance_of(
        self,
        db: &'db dyn Db,
        env: &ProgramEnvironment<'db>,
        typevar: BoundTypeVarIdentity<'db>,
    ) -> VarianceTerm<'db> {
        let constructor = self.constructor(db);
        let Some(arguments) = self.arguments(db) else {
            return VarianceTerm::variable(db, VarianceOrigin::Recursive(constructor), typevar);
        };
        let parameters = arguments.generic_context(db);
        let argument_variances =
            parameters
                .variables(db)
                .zip(arguments.types(db))
                .map(|(parameter, argument)| {
                    argument
                        .variance_of(db, env, typevar)
                        .compose_thunk(db, || {
                            VarianceTerm::variable(
                                db,
                                VarianceOrigin::Recursive(constructor),
                                parameter.identity(db),
                            )
                        })
                });
        let capture_variance = (!parameters
            .variables(db)
            .any(|parameter| parameter.identity(db) == typevar))
        .then(|| VarianceTerm::variable(db, VarianceOrigin::Recursive(constructor), typevar));
        VarianceTerm::join(db, argument_variances.chain(capture_variance))
    }
}

/// Prove that materialization leaves the stored structure unchanged without unfolding a
/// recursive application or evaluating a declaration. A skipped lazy component prevents
/// the proof; it does not justify erasing the pending operation.
fn structurally_static<'db>(db: &'db dyn Db, env: &ProgramEnvironment<'db>, ty: Type<'db>) -> bool {
    struct StaticVisitor<'a, 'db> {
        env: &'a ProgramEnvironment<'db>,
        seen: RefCell<FxHashSet<Type<'db>>>,
        pending: RefCell<Vec<Type<'db>>>,
        is_static: Cell<bool>,
    }

    impl<'db> TypeVisitor<'db> for StaticVisitor<'_, 'db> {
        fn program_environment(&self) -> &ProgramEnvironment<'db> {
            self.env
        }

        fn should_visit_lazy_type_attributes(&self) -> bool {
            false
        }

        fn notify_skipped_lazy_type_attributes(&self) {
            self.is_static.set(false);
        }

        fn visit_type(&self, db: &'db dyn Db, ty: Type<'db>) {
            let _ = db;
            if self.seen.borrow_mut().insert(ty) {
                self.pending.borrow_mut().push(ty);
            }
        }
    }

    impl<'db> StaticVisitor<'_, 'db> {
        fn inspect(&self, db: &'db dyn Db, ty: Type<'db>) {
            if !self.is_static.get() {
                return;
            }
            match ty {
                Type::Dynamic(_) | Type::Divergent(_) => self.is_static.set(false),
                Type::RecursiveVar(variable) => match variable.operations(db).last() {
                    Some(RecursiveOperation::Materialize(_, true)) => {}
                    Some(_) => self.is_static.set(false),
                    None => {
                        if let Some(arguments) = variable.arguments(db) {
                            super::generics::walk_specialization_types(db, arguments, self);
                        }
                    }
                },
                Type::Recursive(recursive) => {
                    if matches!(
                        recursive.operations(db).last(),
                        Some(RecursiveOperation::Materialize(_, true))
                    ) {
                        return;
                    } else if !recursive.operations(db).is_empty() {
                        self.is_static.set(false);
                        return;
                    }
                    if let Some(arguments) = recursive.base_arguments(db) {
                        super::generics::walk_specialization_types(db, arguments, self);
                    }
                    match recursive.body(db) {
                        // A provisional binder supplies no evidence about the final body.
                        RecursiveBody::Bound(Type::RecursiveVar(_)) => self.is_static.set(false),
                        RecursiveBody::Bound(body) => self.visit_type(db, body),
                        _ => self.is_static.set(false),
                    }
                }
                _ => {
                    if let visitor::TypeKind::NonAtomic(ty) = ty.into() {
                        visitor::walk_non_atomic_type(db, ty, self);
                    }
                }
            }
        }
    }

    let visitor = StaticVisitor {
        env,
        seen: RefCell::default(),
        pending: RefCell::default(),
        is_static: Cell::new(true),
    };
    visitor.visit_type(db, ty);
    while visitor.is_static.get() {
        let Some(ty) = visitor.pending.borrow_mut().pop() else {
            break;
        };
        visitor.inspect(db, ty);
    }
    visitor.is_static.get()
}

/// Variables from outside a declaration can remain free in its members. The declaration's
/// own parameters are supplied through its application, so only enclosing scopes contribute
/// captures here. Retaining all enclosing parameters safely includes unused parameters.
pub(super) fn enclosing_type_variables<'db>(
    db: &'db dyn Db,
    definition: Definition<'db>,
) -> impl Iterator<Item = Type<'db>> + 'db {
    let index = semantic_index(db, definition.program_file(db));
    index
        .ancestor_scopes(definition.file_scope(db))
        .filter_map(move |(_, scope)| GenericContext::lexical_of_node(db, scope.node(), index))
        .flat_map(move |context| context.variables(db))
        .filter(move |variable| {
            variable.binding_context(db) != BindingContext::Definition(definition)
        })
        .map(Type::TypeVar)
}

/// Collect variables from the finite representation, without unfolding recursive references.
fn stored_variables<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    ty: Type<'db>,
) -> (FxOrderSet<Type<'db>>, bool) {
    struct VariableVisitor<'a, 'db> {
        env: &'a ProgramEnvironment<'db>,
        seen: RefCell<FxHashSet<Type<'db>>>,
        pending: RefCell<Vec<Type<'db>>>,
        complete: Cell<bool>,
    }

    impl<'db> VariableVisitor<'_, 'db> {
        fn visit_class_application(&self, db: &'db dyn Db, class: ClassType<'db>) {
            if let Some((origin, arguments)) = class.static_class_literal(db) {
                // Class parameters enter through the application. A local declaration can
                // also capture variables from enclosing functions or classes; retain those
                // lexical contexts without forcing inference of its member signatures.
                if let Some(arguments) = arguments {
                    super::generics::walk_specialization_types(db, arguments, self);
                }
                for variable in enclosing_type_variables(db, origin.definition(db)) {
                    self.visit_type(db, variable);
                }
            } else {
                self.complete.set(false);
            }
        }
    }

    impl<'db> TypeVisitor<'db> for VariableVisitor<'_, 'db> {
        fn program_environment(&self) -> &ProgramEnvironment<'db> {
            self.env
        }
        fn should_visit_lazy_type_attributes(&self) -> bool {
            false
        }
        fn notify_skipped_lazy_type_attributes(&self) {
            self.complete.set(false);
        }
        fn visit_protocol_instance_type(
            &self,
            db: &'db dyn Db,
            protocol: ProtocolInstanceType<'db>,
        ) {
            if let Some(class) = protocol.class_origin(db) {
                self.visit_class_application(db, *class);
            } else {
                super::instance::walk_protocol_instance_type(db, protocol, self);
            }
        }
        fn visit_typed_dict_type(&self, db: &'db dyn Db, typed_dict: TypedDictType<'db>) {
            if let Some(class) = typed_dict.defining_class() {
                self.visit_class_application(db, class);
            } else {
                super::typed_dict::walk_typed_dict_type(db, typed_dict, self);
            }
        }
        fn visit_type(&self, _: &'db dyn Db, ty: Type<'db>) {
            if self.seen.borrow_mut().insert(ty) {
                self.pending.borrow_mut().push(ty);
            }
        }
    }

    let visitor = VariableVisitor {
        env,
        seen: RefCell::default(),
        pending: RefCell::new(vec![ty]),
        complete: Cell::new(true),
    };
    let mut variables = FxOrderSet::default();
    loop {
        let Some(ty) = visitor.pending.borrow_mut().pop() else {
            break;
        };
        match ty {
            Type::TypeVar(_) | Type::KnownInstance(super::KnownInstanceType::TypeVar(_)) => {
                variables.insert(ty);
            }
            Type::RecursiveVar(variable) => {
                if let Some(arguments) = variable.arguments(db) {
                    super::generics::walk_specialization_types(db, arguments, &visitor);
                }
                for operation in variable.operations(db) {
                    operation.visit_types(db, &visitor);
                }
            }
            Type::Recursive(recursive) => {
                if let Some(arguments) = recursive.arguments(db) {
                    super::generics::walk_specialization_types(db, arguments, &visitor);
                }
                let (captures, complete) = recursive.constructor(db).captured_variables(db);
                if !*complete {
                    visitor.complete.set(false);
                }
                for &capture in captures {
                    let mut capture = capture;
                    for operation in recursive.operations(db) {
                        operation.with_mapping(|mapping| {
                            let mut operation_visitor =
                                ApplyTypeMappingVisitor::new_for_type_construction(env);
                            if let RecursiveOperation::Materialize(_, map_bounds) = operation {
                                operation_visitor.materialize_typevar_bounds_and_defaults =
                                    *map_bounds;
                            }
                            capture = capture.apply_type_mapping_impl(
                                db,
                                &mapping,
                                TypeContext::default(),
                                &operation_visitor,
                            );
                        });
                    }
                    visitor.visit_type(db, capture);
                }
            }
            Type::TypeAlias(alias) => alias.visit_application_types(db, &visitor),
            _ => {
                if let visitor::TypeKind::NonAtomic(ty) = ty.into() {
                    visitor::walk_non_atomic_type(db, ty, &visitor);
                }
            }
        }
    }
    (variables, visitor.complete.get())
}

impl Type<'_> {
    /// Reject a bare recursive variable at a semantic-operation boundary.
    pub(super) const fn assert_not_recursive_var(self) {
        debug_assert!(
            !matches!(self, Self::RecursiveVar(_)),
            "semantic operation on an unbound recursive variable"
        );
    }
}

#[cfg(test)]
mod tests {
    use ruff_db::files::system_path_to_file;
    use ruff_db::system::DbWithWritableSystem;
    use salsa::plumbing::AsId;
    use ty_python_core::ProgramFile;

    use super::RecursiveType;
    use crate::db::tests::setup_db;
    use crate::place::{Place, global_symbol};
    use crate::types::infer::DefinitionInferenceSource;
    use crate::types::{
        BindingContext, KnownClass, KnownInstanceType, Type, TypeContext, TypeMapping,
    };

    #[test]
    fn inferred_values_promote_nested_singletons() -> anyhow::Result<()> {
        let mut db = setup_db();
        db.write_dedented("/src/a.py", "value = (None,)\n")?;
        let db = &db;
        let env = db.program_environment();
        let file = ProgramFile::new(db, system_path_to_file(db, "/src/a.py")?, env.program(db));
        let Place::Defined(place) = global_symbol(db, file, "value").place else {
            anyhow::bail!("expected value's binding");
        };
        let definition = place
            .provenance
            .definition()
            .ok_or_else(|| anyhow::anyhow!("expected value's definition"))?;
        let reference = RecursiveType::inferred_definition(
            db,
            &env,
            definition.as_id(),
            DefinitionInferenceSource::Definition(definition),
            definition,
            false,
            place.ty,
        );
        assert!(matches!(reference, Type::Recursive(_)));
        let promoted = reference
            .promote_singletons_recursively(db, &env)
            .resolve_inferred_type(db);
        assert_eq!(
            promoted.display(db, &env).to_string(),
            "tuple[None | Unknown]"
        );
        Ok(())
    }

    #[test]
    fn inferred_values_bind_legacy_type_variables() -> anyhow::Result<()> {
        let mut db = setup_db();
        db.write_dedented(
            "/src/a.py",
            "from typing import TypeVar\nT = TypeVar('T', bound=int)\n",
        )?;
        let db = &db;
        let env = db.program_environment();
        let file = ProgramFile::new(db, system_path_to_file(db, "/src/a.py")?, env.program(db));
        let value = global_symbol(db, file, "T").place.expect_type();
        let Type::KnownInstance(KnownInstanceType::TypeVar(typevar)) = value else {
            anyhow::bail!("expected a runtime TypeVar object");
        };
        let definition = typevar
            .definition(db)
            .ok_or_else(|| anyhow::anyhow!("expected T's definition"))?;
        let reference = RecursiveType::inferred_definition(
            db,
            &env,
            definition.as_id(),
            DefinitionInferenceSource::Definition(definition),
            definition,
            false,
            value,
        );
        assert!(matches!(reference, Type::Recursive(_)));
        let context = BindingContext::Definition(definition);
        let bound = reference.apply_type_mapping(
            db,
            &env,
            &TypeMapping::BindLegacyTypevars(context),
            TypeContext::default(),
        );
        let Type::TypeVar(variable) = bound.resolve_inferred_type(db) else {
            anyhow::bail!("binding an inferred value must bind its runtime TypeVar object");
        };
        assert_eq!(variable.binding_context(db), context);
        let rebound = bound.apply_type_mapping(
            db,
            &env,
            &TypeMapping::BindLegacyTypevars(BindingContext::Synthetic(env.program(db))),
            TypeContext::default(),
        );
        assert_eq!(rebound.resolve_inferred_type(db), Type::TypeVar(variable));
        let int = KnownClass::Int.to_instance(db, &env);
        let specialized = bound.substitute_one_typevar(db, &env, variable, int);
        assert_eq!(specialized.resolve_inferred_type(db), int);
        Ok(())
    }
}
