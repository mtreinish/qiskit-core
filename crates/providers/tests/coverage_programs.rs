// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Thise file contains definitions of quantum programs that attempt to cover all features
//! of the `QuantumProgram` IR.
//!
//! - `ops_program` contains every built-in `ProgramOp` at least once.
//! - `tensors_program` contains every tensor type with many shape configurations, and many tensor
//!   values.
//! - `calls_program` contains function calling patterns.
//! - `layout_program` contains various parameters and results locations in a function.
//! - `structures_program` contains various shapes of the input and output data trees.
//!

use std::sync::Arc;

use ndarray::{ArcArrayD, ArrayD, IxDyn, arr0, array, s};
use num_complex::{Complex32, Complex64};
use qiskit_circuit::bit::{ClassicalRegister, QuantumRegister};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Param, StandardGate, StandardInstruction};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr::Symbol;
use qiskit_circuit::{Clbit, Qubit};
use qiskit_providers::ops::{
    Add, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, BroadcastTo, Cast, Constant, Divide,
    Entropy, ErasedProgramOp, Mean, Multiply, Parity, Power, Remainder, ShotLoop, Split, Std,
    Subtract, Variance,
};
use qiskit_providers::tensor::{DType, Dim, KeyTensor, PrngKind, Tensor, TensorType, dim_shape};
use qiskit_providers::{DataTree, FunctionId, InstructionView, ProgramFunction, QuantumProgram};

const DTYPES: [DType; 14] = [
    DType::C128,
    DType::C64,
    DType::F64,
    DType::F32,
    DType::I64,
    DType::I32,
    DType::I16,
    DType::I8,
    DType::U64,
    DType::U32,
    DType::U16,
    DType::U8,
    DType::Bit,
    DType::Key(PrngKind::Philox4x32),
];

/// Return all programs defined in this module.
fn programs() -> Vec<(&'static str, QuantumProgram)> {
    vec![
        ("ops", ops_program()),
        ("tensors", tensors_program()),
        ("calls", calls_program()),
        ("layout", layout_program()),
        ("structures", structures_program()),
    ]
}

/// Return one function per built-in op.
fn op_functions() -> Vec<(&'static str, ProgramFunction)> {
    vec![
        ("add", add()),
        ("subtract", subtract()),
        ("multiply", multiply()),
        ("divide", divide()),
        ("remainder", remainder()),
        ("power", power()),
        ("bitwise_and", bitwise_and()),
        ("bitwise_or", bitwise_or()),
        ("bitwise_xor", bitwise_xor()),
        ("bitwise_not", bitwise_not()),
        ("parity", parity()),
        ("mean", mean()),
        ("variance", variance()),
        ("std", std()),
        ("broadcast_to", broadcast_to()),
        ("cast", cast()),
        ("constant", constant()),
        ("entropy", entropy()),
        ("split", split()),
        ("shot_loop", shot_loop()),
    ]
}

/// Return a program whose entry calls each function of `op_functions` once.
///
/// The entry has a parameter for each parameter of each callee, and a result for each result. The
/// input structure has a sequence per op with a leaf per parameter, and the output structure has a
/// sequence per op with a leaf per result.
fn ops_program() -> QuantumProgram {
    let mut functions = Vec::new();
    let mut entry = ProgramFunction::new();
    let mut inputs = DataTree::new();
    let mut outputs = DataTree::new();
    let leaves =
        |types: &[TensorType]| DataTree::sequence(types.iter().map(|_| DataTree::Leaf(())));
    for (index, (name, function)) in op_functions().into_iter().enumerate() {
        let signature = function.signature();
        let operands: Vec<_> = signature
            .inputs
            .iter()
            .map(|ty| entry.add_parameter(ty.clone()))
            .collect();
        let values = entry
            .add_call(FunctionId::from_index(index), &signature, &operands)
            .unwrap();
        for value in values {
            entry.add_result(value).unwrap();
        }
        inputs
            .insert_branch(name, leaves(&signature.inputs))
            .unwrap();
        outputs
            .insert_branch(name, leaves(&signature.outputs))
            .unwrap();
        functions.push(function);
    }
    functions.push(entry);
    QuantumProgram::new(functions, inputs, outputs).unwrap()
}

// Below there is one function per built-in op. Implementations should attempt to excercise
// every aspect of the data model of the corresponding op, which possibly necessitates
// using the op multiple times.

fn add() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let lhs = f.add_parameter(TensorType {
        dtype: DType::I8,
        shape: dim_shape([2, 1, 2]),
    });
    let rhs = f.add_parameter(TensorType {
        dtype: DType::U8,
        shape: dim_shape([2]),
    });
    for value in f.add_op(Add, &[lhs, rhs]).unwrap() {
        f.add_result(value).unwrap();
    }
    f
}

fn subtract() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let lhs = f.add_parameter(TensorType {
        dtype: DType::I16,
        shape: dim_shape([0]),
    });
    let rhs = f.add_parameter(TensorType {
        dtype: DType::I32,
        shape: dim_shape([1]),
    });
    for value in f.add_op(Subtract, &[lhs, rhs]).unwrap() {
        f.add_result(value).unwrap();
    }
    f
}

fn multiply() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let lhs = f.add_parameter(TensorType {
        dtype: DType::U16,
        shape: dim_shape([2, 2]),
    });
    let rhs = f.add_parameter(TensorType {
        dtype: DType::U32,
        shape: dim_shape([2]),
    });
    for value in f.add_op(Multiply, &[lhs, rhs]).unwrap() {
        f.add_result(value).unwrap();
    }
    f
}

fn divide() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let lhs = f.add_parameter(TensorType {
        dtype: DType::U64,
        shape: dim_shape([1]),
    });
    let rhs = f.add_parameter(TensorType {
        dtype: DType::I64,
        shape: vec![],
    });
    for value in f.add_op(Divide, &[lhs, rhs]).unwrap() {
        f.add_result(value).unwrap();
    }
    f
}

fn remainder() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let lhs = f.add_parameter(TensorType {
        dtype: DType::F64,
        shape: dim_shape([3, 2]),
    });
    let rhs = f.add_parameter(TensorType {
        dtype: DType::F32,
        shape: dim_shape([4, 1, 2]),
    });
    for value in f.add_op(Remainder, &[lhs, rhs]).unwrap() {
        f.add_result(value).unwrap();
    }
    f
}

fn power() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let lhs = f.add_parameter(TensorType {
        dtype: DType::C128,
        shape: dim_shape([3]),
    });
    let rhs = f.add_parameter(TensorType {
        dtype: DType::C64,
        shape: dim_shape([2, 1]),
    });
    for value in f.add_op(Power, &[lhs, rhs]).unwrap() {
        f.add_result(value).unwrap();
    }
    f
}

fn bitwise_and() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let lhs = f.add_parameter(TensorType {
        dtype: DType::Bit,
        shape: vec![Dim::Bounded { max: 8 }, Dim::Fixed(3)],
    });
    let rhs = f.add_parameter(TensorType {
        dtype: DType::Bit,
        shape: dim_shape([3]),
    });
    for value in f.add_op(BitwiseAnd, &[lhs, rhs]).unwrap() {
        f.add_result(value).unwrap();
    }
    f
}

fn bitwise_or() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let lhs = f.add_parameter(TensorType {
        dtype: DType::Bit,
        shape: dim_shape([2, 3]),
    });
    let rhs = f.add_parameter(TensorType {
        dtype: DType::Bit,
        shape: dim_shape([1, 3]),
    });
    for value in f.add_op(BitwiseOr, &[lhs, rhs]).unwrap() {
        f.add_result(value).unwrap();
    }
    f
}

fn bitwise_xor() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let lhs = f.add_parameter(TensorType {
        dtype: DType::Bit,
        shape: vec![],
    });
    let rhs = f.add_parameter(TensorType {
        dtype: DType::Bit,
        shape: dim_shape([4]),
    });
    for value in f.add_op(BitwiseXor, &[lhs, rhs]).unwrap() {
        f.add_result(value).unwrap();
    }
    f
}

fn bitwise_not() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let x = f.add_parameter(TensorType {
        dtype: DType::Bit,
        shape: dim_shape([2, 0]),
    });
    for value in f.add_op(BitwiseNot, &[x]).unwrap() {
        f.add_result(value).unwrap();
    }
    f
}

fn parity() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let x = f.add_parameter(TensorType {
        dtype: DType::Bit,
        shape: vec![Dim::Bounded { max: 5 }, Dim::Fixed(16), Dim::Fixed(2)],
    });
    for value in f.add_op(Parity::new(2), &[x]).unwrap() {
        f.add_result(value).unwrap();
    }
    f
}

fn mean() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let x = f.add_parameter(TensorType {
        dtype: DType::U8,
        shape: dim_shape([3, 1, 16]),
    });
    for value in f.add_op(Mean::new(2), &[x]).unwrap() {
        f.add_result(value).unwrap();
    }
    f
}

fn variance() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let x = f.add_parameter(TensorType {
        dtype: DType::F32,
        shape: dim_shape([4, 2]),
    });
    for value in f.add_op(Variance::new(0, 1.0), &[x]).unwrap() {
        f.add_result(value).unwrap();
    }
    f
}

fn std() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let x = f.add_parameter(TensorType {
        dtype: DType::C64,
        shape: dim_shape([2, 1]),
    });
    for value in f.add_op(Std::new(1, 0.5), &[x]).unwrap() {
        f.add_result(value).unwrap();
    }
    f
}

fn broadcast_to() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let x = f.add_parameter(TensorType {
        dtype: DType::F32,
        shape: vec![Dim::Bounded { max: 8 }],
    });
    let target = vec![Dim::Fixed(4), Dim::Bounded { max: 8 }];
    for value in f.add_op(BroadcastTo::new(target), &[x]).unwrap() {
        f.add_result(value).unwrap();
    }
    f
}

fn cast() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let x = f.add_parameter(TensorType {
        dtype: DType::Bit,
        shape: dim_shape([2]),
    });
    // all types Bit can be cast to
    let dtypes = [
        DType::C128,
        DType::C64,
        DType::F64,
        DType::F32,
        DType::I64,
        DType::I32,
        DType::I16,
        DType::I8,
        DType::U64,
        DType::U32,
        DType::U16,
        DType::U8,
        DType::Bit,
    ];
    for dtype in dtypes {
        for value in f.add_op(Cast::new(dtype), &[x]).unwrap() {
            f.add_result(value).unwrap();
        }
    }
    f
}

fn constant() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let op = Constant::new(Tensor::from([1.0_f64, 2.0]));
    for value in f.add_op(op, &[]).unwrap() {
        f.add_result(value).unwrap();
    }
    f
}

fn entropy() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    for shape in [vec![], vec![2, 2]] {
        let op = Entropy::new(PrngKind::Philox4x32, shape);
        for value in f.add_op(op, &[]).unwrap() {
            f.add_result(value).unwrap();
        }
    }
    f
}

fn split() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let scalar = f.add_parameter(TensorType {
        dtype: DType::Key(PrngKind::Philox4x32),
        shape: vec![],
    });
    let vector = f.add_parameter(TensorType {
        dtype: DType::Key(PrngKind::Philox4x32),
        shape: dim_shape([2]),
    });
    for (key, shape) in [(scalar, vec![2, 3]), (vector, vec![3, 1])] {
        for value in f.add_op(Split::new(shape), &[key]).unwrap() {
            f.add_result(value).unwrap();
        }
    }
    f
}

fn shot_loop() -> ProgramFunction {
    let mut f = ProgramFunction::new();
    let with_parameters = f.add_parameter(TensorType {
        dtype: DType::F64,
        shape: vec![Dim::Bounded { max: 5 }, Dim::Fixed(3)],
    });
    let without_parameters = f.add_parameter(TensorType {
        dtype: DType::F32,
        shape: dim_shape([3, 1, 0]),
    });
    let circuits = vec![circuit_with_parameters(), circuit_without_parameters()];
    let op = ShotLoop::new(circuits, 16).unwrap();
    let values = f.add_op(op, &[with_parameters, without_parameters]);
    for value in values.unwrap() {
        f.add_result(value).unwrap();
    }
    f
}

/// Return a circuit of three parameters and two classical registers.
fn circuit_with_parameters() -> CircuitData {
    let mut circuit = CircuitData::new(None, None, Param::Float(0.0)).unwrap();
    circuit
        .add_qreg(QuantumRegister::new_owning("q", 2), true)
        .unwrap();
    circuit
        .add_creg(ClassicalRegister::new_owning("c", 2), true)
        .unwrap();
    circuit
        .add_creg(ClassicalRegister::new_owning("flags", 1), true)
        .unwrap();
    for (gate, qubit, name) in [
        (StandardGate::RX, 0, "a0"),
        (StandardGate::RY, 1, "a1"),
        (StandardGate::RZ, 0, "a2"),
    ] {
        let angle = ParameterExpression::from_symbol(Symbol::standalone(name.to_string(), None));
        let angle = Param::ParameterExpression(Arc::new(angle));
        circuit
            .push_standard_gate(gate, &[angle], &[Qubit(qubit)])
            .unwrap();
    }
    for (qubit, clbit) in [(0, 0), (1, 1), (1, 2)] {
        circuit
            .push_packed_operation(
                PackedOperation::from_standard_instruction(StandardInstruction::Measure),
                None,
                &[Qubit(qubit)],
                &[Clbit(clbit)],
            )
            .unwrap();
    }
    circuit
}

/// Return a circuit of no parameters and one classical register.
fn circuit_without_parameters() -> CircuitData {
    let mut circuit = CircuitData::new(None, None, Param::Float(0.0)).unwrap();
    circuit
        .add_qreg(QuantumRegister::new_owning("q", 3), true)
        .unwrap();
    circuit
        .add_creg(ClassicalRegister::new_owning("meas", 3), true)
        .unwrap();
    for bit in 0..3 {
        circuit
            .push_standard_gate(StandardGate::H, &[], &[Qubit(bit)])
            .unwrap();
        circuit
            .push_packed_operation(
                PackedOperation::from_standard_instruction(StandardInstruction::Measure),
                None,
                &[Qubit(bit)],
                &[Clbit(bit)],
            )
            .unwrap();
    }
    circuit
}

/// Return a program covering every tensor type and a range of tensor values.
///
/// The one function has a parameter of each dtype in each shape, and a constant of each value.
/// Every parameter and every constant is also a result.
///
/// There is a value of each dtype. One matrix is in row-major order and two are not. One value
/// is a scalar, one has no elements, and one has an axis too long for 32 bits.
fn tensors_program() -> QuantumProgram {
    let shapes = [
        vec![],
        dim_shape([2, 0]),
        dim_shape([1, 2, 1, 1, 2, 1]),
        dim_shape([usize::MAX]),
        // Two bounded axes, with a fixed axis between them.
        vec![
            Dim::Bounded { max: 4 },
            Dim::Fixed(3),
            Dim::Bounded { max: 8 },
        ],
        // A bounded axis between two fixed axes.
        vec![Dim::Fixed(2), Dim::Bounded { max: 4 }, Dim::Fixed(3)],
        vec![Dim::Bounded { max: 0 }],
        vec![Dim::Bounded { max: 1 }, Dim::Fixed(2)],
        vec![Dim::Bounded { max: usize::MAX }],
    ];
    // The view starts partway into its buffer and steps backwards along its last axis. Its values
    // are `[[u16::MAX, 0], [2, 1]]`.
    let reversed = array![[10_u16, 11], [0, u16::MAX], [1, 2]]
        .into_shared()
        .slice_move(s![1.., ..;-1])
        .into_dyn();
    // An array has no elements when any axis is empty, so another axis may be as long as `isize`
    // allows. On a 64-bit target, that length does not fit in 32 bits.
    let longest_axis = ArrayD::<u8>::from_shape_vec(IxDyn(&[0, isize::MAX as usize]), vec![]);
    let words = array![[0, u32::MAX], [0xdead_beef, 1]]
        .into_dyn()
        .into_shared();
    let tensors = [
        Tensor::from([
            Complex64::new(f64::NAN, -0.0),
            Complex64::new(f64::INFINITY, f64::NEG_INFINITY),
            Complex64::new(5e-324, f64::MAX),
        ]),
        Tensor::from([Complex32::new(1.5, -2.0), Complex32::new(-0.0, f32::NAN)]),
        // Column-major.
        Tensor::from(
            array![
                [f64::NAN, -0.0, f64::INFINITY],
                [f64::NEG_INFINITY, 5e-324, f64::MAX]
            ]
            .reversed_axes()
            .into_dyn(),
        ),
        Tensor::from([
            1.0_f32,
            -0.0,
            f32::MIN_POSITIVE,
            1e-45,
            f32::MAX,
            f32::NAN,
            -1.5,
            3.25,
        ]),
        Tensor::from(arr0(i64::MIN).into_dyn()),
        Tensor::from([i32::MIN, -1, 0, i32::MAX]),
        Tensor::from(&[] as &[i16]),
        Tensor::from(array![[i8::MIN, i8::MAX], [-1, 0]].into_dyn()),
        Tensor::from([u64::MAX]),
        Tensor::from([u32::MAX, 7]),
        Tensor::from(reversed),
        Tensor::from([0_u8, u8::MAX]),
        Tensor::from(longest_axis.unwrap()),
        Tensor::Bit(array![1, 0, 1].into_dyn().into_shared()),
        Tensor::Bit(arr0(1).into_dyn().into_shared()),
        Tensor::Key(KeyTensor::new(PrngKind::Philox4x32, words).unwrap()),
    ];

    let mut f = ProgramFunction::new();
    for dtype in DTYPES {
        for shape in &shapes {
            let x = f.add_parameter(TensorType {
                dtype,
                shape: shape.clone(),
            });
            f.add_result(x).unwrap();
        }
    }
    for tensor in tensors {
        for value in f.add_op(Constant::new(tensor), &[]).unwrap() {
            f.add_result(value).unwrap();
        }
    }
    let signature = f.signature();
    let leaves = |count: usize| DataTree::sequence((0..count).map(|_| DataTree::Leaf(())));
    let inputs = leaves(signature.inputs.len());
    let outputs = leaves(signature.outputs.len());
    QuantumProgram::new(vec![f], inputs, outputs).unwrap()
}

/// Return a program covering calls between functions. It has no ops.
///
/// - `@0` has no parameters and no results.
/// - `@1` is never called.
/// - `@2` has a parameter with a bounded axis.
/// - `@3` calls `@2` with an operand whose axis has a fixed size within the bound.
/// - `@4` is the entry. It calls `@0`, then `@3`, then `@2` twice.
fn calls_program() -> QuantumProgram {
    let empty = ProgramFunction::new();

    let mut uncalled = ProgramFunction::new();
    let x = uncalled.add_parameter(TensorType {
        dtype: DType::I64,
        shape: dim_shape([1]),
    });
    uncalled.add_result(x).unwrap();

    let mut bounded = ProgramFunction::new();
    let x = bounded.add_parameter(TensorType {
        dtype: DType::F64,
        shape: vec![Dim::Bounded { max: 4 }],
    });
    bounded.add_result(x).unwrap();

    let fixed = TensorType {
        dtype: DType::F64,
        shape: dim_shape([3]),
    };
    let mut caller = ProgramFunction::new();
    let x = caller.add_parameter(fixed.clone());
    let values = caller
        .add_call(FunctionId::from_index(2), &bounded.signature(), &[x])
        .unwrap();
    for value in values {
        caller.add_result(value).unwrap();
    }

    let mut entry = ProgramFunction::new();
    let x = entry.add_parameter(fixed);
    entry
        .add_call(FunctionId::from_index(0), &empty.signature(), &[])
        .unwrap();
    for (callee, signature) in [
        (3, caller.signature()),
        (2, bounded.signature()),
        (2, bounded.signature()),
    ] {
        let values = entry
            .add_call(FunctionId::from_index(callee), &signature, &[x])
            .unwrap();
        for value in values {
            entry.add_result(value).unwrap();
        }
    }

    QuantumProgram::new(
        vec![empty, uncalled, bounded, caller, entry],
        DataTree::sequence([DataTree::Leaf(())]),
        DataTree::sequence((0..3).map(|_| DataTree::Leaf(()))),
    )
    .unwrap()
}

/// Return a program covering where parameters and results may appear in a function.
fn layout_program() -> QuantumProgram {
    let mut f = ProgramFunction::new();
    let a = f.add_parameter(TensorType {
        dtype: DType::F64,
        shape: dim_shape([2]),
    });
    let mean = f.add_op(Mean::new(0), &[a]).unwrap()[0];
    // A result before a later op.
    f.add_result(mean).unwrap();
    // A parameter after an op.
    let b = f.add_parameter(TensorType {
        dtype: DType::F64,
        shape: dim_shape([2]),
    });
    // An op whose output is never used.
    f.add_op(Add, &[a, b]).unwrap();
    // The same value as a result twice, and a parameter as a result.
    f.add_result(mean).unwrap();
    f.add_result(b).unwrap();

    QuantumProgram::new(
        vec![f],
        DataTree::sequence([DataTree::Leaf(()), DataTree::Leaf(())]),
        DataTree::sequence((0..3).map(|_| DataTree::Leaf(()))),
    )
    .unwrap()
}

/// Return a program covering the shapes of a data tree.
///
/// The input structure is a single leaf. The output structure has nested sequences and mappings,
/// empty branches, a branch with both positional and named children, and a name outside ASCII. The
/// function returns its parameter once for each output leaf.
fn structures_program() -> QuantumProgram {
    let mut mixed = DataTree::new();
    mixed.push_leaf(());
    mixed.insert_leaf("named", ()).unwrap();
    mixed.push_leaf(());
    let outputs = DataTree::mapping([
        ("leaf", DataTree::Leaf(())),
        ("θ", DataTree::Leaf(())),
        (
            "sequence",
            DataTree::sequence([
                DataTree::sequence([DataTree::Leaf(())]),
                DataTree::mapping([("leaf", DataTree::Leaf(()))]).unwrap(),
                DataTree::new(),
                DataTree::Leaf(()),
            ]),
        ),
        ("mixed", mixed),
        ("empty", DataTree::new()),
    ])
    .unwrap();

    let mut f = ProgramFunction::new();
    let x = f.add_parameter(TensorType {
        dtype: DType::U8,
        shape: dim_shape([2]),
    });
    for _ in 0..outputs.leaf_count() {
        f.add_result(x).unwrap();
    }
    QuantumProgram::new(vec![f], DataTree::Leaf(()), outputs).unwrap()
}

#[test]
fn test_programs_are_equal() {
    // sanity check on assert_same_program
    for ((name, actual), (_, expected)) in programs().into_iter().zip(programs()) {
        assert_same_program(name, &actual, &expected);
    }
}

/// Assert that two programs are the same, instruction by instruction.
fn assert_same_program(name: &str, actual: &QuantumProgram, expected: &QuantumProgram) {
    assert_eq!(
        actual.input_structure(),
        expected.input_structure(),
        "{name}: input structure"
    );
    assert_eq!(
        actual.output_structure(),
        expected.output_structure(),
        "{name}: output structure"
    );
    assert_eq!(
        actual.functions().len(),
        expected.functions().len(),
        "{name}: function count"
    );
    for (index, (actual, expected)) in actual
        .functions()
        .iter()
        .zip(expected.functions())
        .enumerate()
    {
        assert_same_function(actual, expected, &format!("{name} @{index}"));
    }
}

fn assert_same_function(actual: &ProgramFunction, expected: &ProgramFunction, at: &str) {
    assert_eq!(actual.signature(), expected.signature(), "{at}: signature");
    assert_eq!(
        actual.parameters(),
        expected.parameters(),
        "{at}: parameters"
    );
    assert_eq!(actual.results(), expected.results(), "{at}: results");
    assert_eq!(
        actual.instruction_count(),
        expected.instruction_count(),
        "{at}: instruction count"
    );
    for (actual, expected) in actual.iter_instructions().zip(expected.iter_instructions()) {
        let at = format!("{at} %{}", expected.id());
        assert_eq!(actual.full_name(), expected.full_name(), "{at}: name");
        assert_eq!(actual.role(), expected.role(), "{at}: role");
        assert_eq!(actual.operands(), expected.operands(), "{at}: operands");
        assert_eq!(
            actual.output_types(),
            expected.output_types(),
            "{at}: output types"
        );
        assert_eq!(
            actual.has_builtin_eval(),
            expected.has_builtin_eval(),
            "{at}: builtin eval"
        );
        match (actual.view(), expected.view()) {
            (InstructionView::Call(actual), InstructionView::Call(expected)) => {
                assert_eq!(actual, expected, "{at}: callee")
            }
            (InstructionView::Op(actual), InstructionView::Op(expected)) => {
                assert_same_op(actual, expected, &at)
            }
            _ => {}
        }
    }
}

fn assert_same_op(
    actual: &(dyn ErasedProgramOp + 'static),
    expected: &(dyn ErasedProgramOp + 'static),
    at: &str,
) {
    if let Some(expected) = expected.downcast_ref::<Constant>() {
        let actual = actual.downcast_ref::<Constant>().unwrap();
        assert!(
            same_tensor(actual.value(), expected.value()),
            "{at}: {:?} differs from {:?}",
            actual.value(),
            expected.value()
        );
    } else if let Some(expected) = expected.downcast_ref::<ShotLoop>() {
        let actual = actual.downcast_ref::<ShotLoop>().unwrap();
        assert_eq!(actual.shots(), expected.shots(), "{at}: shots");
        assert_eq!(
            actual.circuits().len(),
            expected.circuits().len(),
            "{at}: circuit count"
        );
        for (index, (actual, expected)) in actual
            .circuits()
            .iter()
            .zip(expected.circuits())
            .enumerate()
        {
            assert_same_circuit_io(actual, expected, &format!("{at} circuit {index}"));
        }
    } else {
        // Every other op is a few plain fields, all of which are in its `Debug` output. Not every
        // op has an accessor for each field.
        assert_eq!(
            format!("{actual:?}"),
            format!("{expected:?}"),
            "{at}: payload"
        );
    }
}

/// Return whether two tensors have the same dtype, shape and element bits.
fn same_tensor(actual: &Tensor, expected: &Tensor) -> bool {
    fn same<T, U: PartialEq>(a: &ArcArrayD<T>, b: &ArcArrayD<T>, bits: impl Fn(&T) -> U) -> bool {
        a.shape() == b.shape() && a.iter().map(&bits).eq(b.iter().map(&bits))
    }
    match (actual, expected) {
        (Tensor::C64(a), Tensor::C64(b)) => same(a, b, |z| (z.re.to_bits(), z.im.to_bits())),
        (Tensor::C128(a), Tensor::C128(b)) => same(a, b, |z| (z.re.to_bits(), z.im.to_bits())),
        (Tensor::F32(a), Tensor::F32(b)) => same(a, b, |x| x.to_bits()),
        (Tensor::F64(a), Tensor::F64(b)) => same(a, b, |x| x.to_bits()),
        (Tensor::I8(a), Tensor::I8(b)) => same(a, b, |&x| x),
        (Tensor::I16(a), Tensor::I16(b)) => same(a, b, |&x| x),
        (Tensor::I32(a), Tensor::I32(b)) => same(a, b, |&x| x),
        (Tensor::I64(a), Tensor::I64(b)) => same(a, b, |&x| x),
        (Tensor::U8(a), Tensor::U8(b)) => same(a, b, |&x| x),
        (Tensor::U16(a), Tensor::U16(b)) => same(a, b, |&x| x),
        (Tensor::U32(a), Tensor::U32(b)) => same(a, b, |&x| x),
        (Tensor::U64(a), Tensor::U64(b)) => same(a, b, |&x| x),
        (Tensor::Bit(a), Tensor::Bit(b)) => same(a, b, |&x| x),
        (Tensor::Key(a), Tensor::Key(b)) => {
            a.kind() == b.kind() && same(a.words(), b.words(), |&x| x)
        }
        _ => false,
    }
}

/// Assert that two circuits agree in what the program depends on.
fn assert_same_circuit_io(actual: &CircuitData, expected: &CircuitData, at: &str) {
    assert_eq!(
        actual.num_parameters(),
        expected.num_parameters(),
        "{at}: parameters"
    );
    assert_eq!(
        classical_registers(actual),
        classical_registers(expected),
        "{at}: classical registers"
    );
    assert_eq!(actual.num_qubits(), expected.num_qubits(), "{at}: qubits");
    assert_eq!(
        actual.data().len(),
        expected.data().len(),
        "{at}: instruction count"
    );
}

/// Return the name and width of each classical register.
fn classical_registers(circuit: &CircuitData) -> Vec<(&str, usize)> {
    circuit
        .cregs()
        .iter()
        .map(|register| (register.name(), register.len()))
        .collect()
}
