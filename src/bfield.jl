using NIfTI

function bfield_correction(image_path, mask_path)
	inputImage = sitk.ReadImage(image_path, sitk.sitkFloat32)
	image = inputImage
	
	maskImage = sitk.ReadImage(mask_path, sitk.sitkUInt8)
	
	corrector = sitk.N4BiasFieldCorrectionImageFilter()

	corrected_image = corrector.Execute(image, maskImage)
    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)

	tempdir = mktempdir()
	corrected_image_path = joinpath(tempdir, "corrected_image.nii")
	log_bias_field_path = joinpath(tempdir, "log_bias_field.nii")

	sitk.WriteImage(corrected_image, corrected_image_path)
	sitk.WriteImage(log_bias_field, log_bias_field_path)

	return (
		niread(image_path),
		niread(mask_path),
		niread(log_bias_field_path),
		niread(corrected_image_path)
	)
end