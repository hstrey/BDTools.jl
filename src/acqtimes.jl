import JSON3
using DelimitedFiles

function acqtimes(jsonfilename)
    json_string = read(jsonfilename, String)
    data = JSON3.read(json_string)
    acqt = Int64.(round.(Float64.(data["SliceTiming"])*1000))
    slices = collect(1:length(acqt))
    return slices,acqt
end

function write_acqtimes(slices,acqt,outfile)
    io = open(outfile, "w")
    write(io,"Time,Slice\n")
    for i in 1:length(acqt)
        write(io, string(acqt[i])*","*string(slices[i])*"\n")
    end
    close(io)
end

