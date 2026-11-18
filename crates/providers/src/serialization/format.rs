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

use binrw::{BinRead, BinResult, BinWrite, Endian, VecArgs, binread, binrw, binwrite};
use num_complex::{Complex32, Complex64};
use qiskit_qpy::Bytes;

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ProgramHeader {
    #[brw(magic = b"QKPROG")]
    pub serialization_version: u8,
    pub qiskit_version: (u8, u8, u8),
    pub input_structure: DataTreePack,
    pub output_structure: DataTreePack,
    #[bw(calc = functions.len() as u64)]
    pub num_functions: u64,
    #[br(args((num_functions,)), parse_with=functions_decoder)]
    pub functions: Vec<ProgramFunctionPack>,
}

#[binrw::parser(reader, endian)]
fn functions_decoder(args: (u64,)) -> BinResult<Vec<ProgramFunctionPack>> {
    let count = args.0 as usize;
    let mut out = Vec::new();
    out.try_reserve_exact(count).map_err(|e| {
        binrw::Error::Io(binrw::io::Error::new(
            binrw::io::ErrorKind::OutOfMemory,
            e.to_string(),
        ))
    })?;
    for _ in 0..count {
        out.push(<_>::read_options(reader, endian, ())?);
    }
    Ok(out)
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ProgramFunctionPack {
    #[bw(calc = instructions.len() as u32)]
    num_instructions: u32,
    #[bw(calc = parameters.len() as u32)]
    num_parameters: u32,
    #[bw(calc = results.len() as u32)]
    num_results: u32,
    #[br(count = num_instructions)]
    instructions: Vec<ProgramInstructionPack>,
    #[br(count = num_parameters)]
    parameters: Vec<u32>,
    #[br(count = num_results)]
    results: Vec<u32>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ProgramInstructionPack {
    body: InstructionBodyPack,
    #[bw(calc = operands.len() as u32)]
    num_operands: u32,
    #[bw(calc = output_types.len() as u32)]
    num_output_types: u32,
    #[br(count = num_operands)]
    operands: Vec<ValuePack>,
    #[br(count = num_output_types)]
    output_types: Vec<TensorTypePack>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
struct ValuePack {
    instruction: u32,
    slot: u16,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
struct TensorTypePack {
    dtype: DtypePack,
    #[bw(calc = shape.len() as u32)]
    shape_len: u32,
    #[br(count = shape_len)]
    shape: Vec<DimPack>
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
enum DimPack {
    #[brw(magic = b'f')]
    Fixed(u64),
    #[brw(magic = b'b')]
    Bounded(u64),
}

#[binrw]
#[brw(big)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum DtypePack {
    #[brw(magic = b'C')]
    C128, // complex
    #[brw(magic = b'c')]
    C64,
    #[brw(magic = b'F')]
    F64, // real
    #[brw(magic = b'f')]
    F32,
    #[brw(magic = b'I')]
    I64, // signed integer
    #[brw(magic = b'i')]
    I32,
    #[brw(magic = b'l')]
    I16,
    #[brw(magic = b'b')]
    I8,
    #[brw(magic = b'U')]
    U64, // unsigned integer
    #[brw(magic = b'u')]
    U32,
    #[brw(magic = b'L')]
    U16,
    #[brw(magic = b'B')]
    U8,
    #[brw(magic = b'n')]
    Bit,           // bool
    #[brw(magic = b'P')]
    Philo4x32,
}


#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum InstructionBodyPack {
    #[brw(magic = b'p')]
    Parameter,
    #[brw(magic = b'o')]
    Op(ProgramOpPack),
    #[brw(magic = b'c')]
    Call(u32),
    #[brw(magic = b'r')]
    Result,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
enum ProgramOpPack {
    #[brw(magic = b'+')]
    Add,
    #[brw(magic = b'/')]
    Divide,
    #[brw(magic = b'x')]
    Multiply,
    #[brw(magic = b'*')]
    Power,
    #[brw(magic = b'r')]
    Remainder,
    #[brw(magic = b's')]
    Subtract,
    #[brw(magic = b'&')]
    BitwiseAnd,
    #[brw(magic = b'!')]
    BitwiseNot,
    #[brw(magic = b'|')]
    BitwiseOr,
    #[brw(magic = b'^')]
    BitwiseXor,
    #[brw(magic = b'p')]
    Pairity(u64),
    #[brw(magic = b'b')]
    BroadcastTo {
        #[bw(calc = targets.len() as u64)]
        num_targets: u64,
        #[br(count = num_targets)]
        targets: Vec<DimPack>,
    },
    #[brw(magic = b'c')]
    Cast(DtypePack),
    #[brw(magic = b'C')]
    Constant(TensorPack),
    #[brw(magic = b'm')]
    Mean(u64),
    #[brw(magic = b'S')]
    Std(u64, f64),
    #[brw(magic = b'v')]
    Variance(u64, f64),
    #[brw(magic = b'L')]
    ShotLoop(ShotLoopPack),
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
struct ShotLoopPack {
    shots: u64,
    #[bw(calc = circuits.len() as u64)]
    circuits_size: u64,
    #[br(count = circuits_size)]
    circuits: Bytes,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
struct TensorPack {
    dtype: DtypePack,
    #[bw(calc = shape.len() as u32)]
    shape_len: u32,
    #[br(count = shape_len)]
    shape: Vec<u64>,
    num_elems: u64,
    #[br(args(dtype, num_elems as usize))]
    data: TensorData,
}

#[binrw]
#[brw(big)]
#[br(import(dtype: DtypePack, len: usize))]
#[derive(Debug)]
enum TensorData {
    #[br(pre_assert(dtype == DtypePack::C128))]
    C128(#[br(count = len)] Vec<[f64;2]>),
    #[br(pre_assert(dtype == DtypePack::C64))]
    C64(#[br(count = len)] Vec<[f32;2]>),
    #[br(pre_assert(dtype == DtypePack::F64))]
    F64(#[br(count = len)] Vec<f64>),
    #[br(pre_assert(dtype == DtypePack::F32))]
    F32(#[br(count = len)] Vec<f32>),
    #[br(pre_assert(dtype == DtypePack::I64))]
    I64(#[br(count = len)] Vec<i64>),
    #[br(pre_assert(dtype == DtypePack::I32))]
    I32(#[br(count = len)] Vec<i32>),
    #[br(pre_assert(dtype == DtypePack::I16))]
    I16(#[br(count = len)] Vec<i16>),
    #[br(pre_assert(dtype == DtypePack::I8))]
    I8(#[br(count = len)] Vec<i8>),
    #[br(pre_assert(dtype == DtypePack::U64))]
    U64(#[br(count = len)] Vec<u64>),
    #[br(pre_assert(dtype == DtypePack::U32))]
    U32(#[br(count = len)] Vec<u32>),
    #[br(pre_assert(dtype == DtypePack::U16))]
    U16(#[br(count = len)] Vec<u16>),
    #[br(pre_assert(dtype == DtypePack::U8))]
    U8(#[br(count = len)] Vec<u8>),
    #[br(pre_assert(dtype == DtypePack::Bit))]
    Bit(#[br(count = len)] Vec<u8>),
    #[br(pre_assert(dtype == DtypePack::Philo4x32))]
    Philo4x32(#[br(count = len)] Vec<u32>),
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
struct DataTreePack {
    // TODO
}
