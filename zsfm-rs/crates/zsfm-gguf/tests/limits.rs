use std::io::Cursor;
use zsfm_gguf::{GGMLType, GGUFFile, GGUFMetaValue, GGUFWriter};

#[test]
fn rejects_bad_magic() {
    let mut buf = Cursor::new(b"NOPE".to_vec());
    assert!(GGUFFile::read(&mut buf).is_err());
}

#[test]
fn rejects_huge_string() {
    use byteorder::{LittleEndian, WriteBytesExt};
    let mut raw = Vec::new();
    raw.extend_from_slice(b"GGUF");
    raw.write_u32::<LittleEndian>(3).unwrap();
    raw.write_u64::<LittleEndian>(0).unwrap(); // tensor_count
    raw.write_u64::<LittleEndian>(1).unwrap(); // kv_count
    raw.write_u64::<LittleEndian>(20 << 20).unwrap(); // 20 MiB string (over 16 MiB cap)
    let mut buf = Cursor::new(raw);
    assert!(GGUFFile::read(&mut buf).is_err());
}

#[test]
fn rejects_huge_counts() {
    use byteorder::{LittleEndian, WriteBytesExt};
    let mut raw = Vec::new();
    raw.extend_from_slice(b"GGUF");
    raw.write_u32::<LittleEndian>(3).unwrap();
    raw.write_u64::<LittleEndian>(2_000_000).unwrap(); // tensor_count over cap
    raw.write_u64::<LittleEndian>(0).unwrap();
    let mut buf = Cursor::new(raw);
    assert!(GGUFFile::read(&mut buf).is_err());
}

#[test]
fn roundtrip_then_tensor_bytes_cap() {
    let mut w = GGUFWriter::new();
    w.add_metadata("general.architecture", GGUFMetaValue::String("test".into()));
    let vals: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    w.add_tensor("t", vec![32], GGMLType::F32, bytes);
    let mut buf = Cursor::new(Vec::new());
    w.write_to(&mut buf).unwrap();
    buf.set_position(0);
    let f = GGUFFile::read(&mut buf).unwrap();
    assert_eq!(f.tensors.len(), 1);
    let data = f.tensor_f32(&mut buf, &f.tensors[0]).unwrap();
    assert_eq!(data.len(), 32);
}
