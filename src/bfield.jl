using NIfTI

function bfield_correction(image_path, mask_path; spline_order=3, num_control_points=[4, 4, 4])
    tempdir = mktempdir()
    corrected_image_path = joinpath(tempdir, "corrected_image.nii")
    bias_field_path = joinpath(tempdir, "bias_field.nii")

    sys_path = ""
    if Sys.islinux()
        sys_path = "linux"
    elseif Sys.isapple()
        sys_path = "apple"
    elseif Sys.iswindows()
        sys_path = "windows"
    else
        throw("current operating system unsupported")
    end

    N4Bias_path = joinpath("ANTs",sys_path,"N4BiasFieldCorrection")
    ANTscommand = `$N4Bias_path -s 1 -d 3 -b \[1x1x1, 3\] -i $image_path - x $mask_path -o \[ $corrected_image_path , $bias_field_path \]`

    return (
        niread(image_path),
        niread(mask_path),
        niread(bias_field_path),
        niread(corrected_image_path)
    )
end