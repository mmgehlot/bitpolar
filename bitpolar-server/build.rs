fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("proto/bitpolar/v1/service.proto")?;
    Ok(())
}
