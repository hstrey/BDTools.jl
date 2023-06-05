using NIfTI

function bfield_correction(image_path, mask_path; spline_order=3, num_control_points=[4, 4, 4])
    spline_order = pyint(spline_order)
    num_control_points = pylist(num_control_points)

    inputImage = sitk.ReadImage(image_path, sitk.sitkFloat32)
    image = inputImage

    # Check that num_control_points is the same dimension as the image
    sz = image.GetSize()
    @assert(
		length(pyconvert(Vector, sz)) == length(pyconvert(Vector, num_control_points)), 
		"Incorrect size for number of control points, make sure the vector matches the dimensions of the image"
	)

    maskImage = sitk.ReadImage(mask_path, sitk.sitkUInt8)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetSplineOrder(spline_order)
    corrector.SetNumberOfControlPoints(num_control_points)

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