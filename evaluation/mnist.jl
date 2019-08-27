using Flux, Flux.Data.MNIST, SumDenseProduct, IterTools, StatsBase, Distributions, Unitary, Serialization
using Images
using Base.Iterators: partition
using Flux: throttle, train!, Params
using EvalCurves, DrWatson, BSON
using SumDenseProduct: samplepath

using Plots
plotly();
showimg(x) = heatmap(reshape(x, round(Int,sqrt(49)),round(Int,sqrt(49))))



# imgs = MNIST.images();
# imgs = map(x -> imresize(x, ratio = 1/4), imgs);
# X = Float32.(float(hcat(vec.(imgs)...)));
# d = size(X,1)
# noise = [0.25,  0]
# nc = [10, 10]
# model = buildmixture(d, nc,fill(identity, length(nc)), round.(Int,noise.*d), sharing = :all);
# xval = X[:,sample(1:size(X,2),100, replace = false)];
# batchsize, iterations, maxpath = 100, 20000, 100
# SumDenseProduct.updatelatent!(model, xval, 32)
# SumDenseProduct.tunegrad(model, X, batchsize, maxpath, Flux.params(model), [:sampling, :exactpath])


function experiment()
	imgs = MNIST.images();
	imgs = map(x -> imresize(x, ratio = 1/4), imgs);
	X = Float32.(float(hcat(vec.(imgs)...)));
	d = size(X,1)
	odir = "/mnt/output/results/datasets/mnist/sumdense"
	noise = 0
	for l in [2,3,4]
		for k in [10,20,40]
			for noise in [fill(1/l, l), fill(0, l)]
				nc = fill(k, l)
				println("training ",nc," ", noise)
				ofile = "exactpath_ratio=4_nc="*join(nc,"-")*"_noise="*join(noise,"-")
				if isfile(joinpath(odir, ofile*"_model.jls"))
					println("skipping")
					continue
				end
				model = buildmixture(d, nc,fill(identity, length(nc)), round.(Int,noise.*d));
				tstidx = sample(1:size(X,2),100, replace = false)
				xval = X[:,tstidx]

				batchsize, iterations, maxpath = 100, 20000, 100
				fit!(model, X, batchsize, iterations, maxpath, xval = xval, gradmethod = :exactpath);
				serialize(joinpath(odir, ofile*"_model.jls"),model)
				o = SumDenseProduct.batchlogpdf(model, X, 100)
				serialize(joinpath(odir, ofile*"_logpdf.jls"),o)
			end
		end
	end
end
experiment()


function show()
	odir = "/mnt/output/results/datasets/mnist/sumdense"
	foreach(filter(s -> endswith(s,"_logpdf.jls"),readdir(odir))) do f 
		o = deserialize(joinpath(odir, f));
		s = replace(f, "exactpath_ratio=4_" => "")
		s = replace(s, "_logpdf.jls" => "")
		println(mean(o), "  ",s)
	end
end

function showsamples()
	mdir = "/Users/tpevny/Work/Julia/results/datasets/mnist/sumdense/"
	f = "exactpath_ratio=4_nc=10-10-10_noise=0.0-0.0-0.0"
	println("logpdf: ",mean(deserialize(joinpath(mdir, f*"_logpdf.jls" ))))

	model = deserialize(joinpath(mdir, f*"_model.jls"))
end

#noise, nc = [0.25, 0.25, 0.25, 0], [10,10,10, 10], ratio 1/4
# 								16 threads		1 thread
# compilation of samplinggrad: 	0.962394178		32.596927812
# execution of samplinggrad: 	1.002392203		1.377766636
# compilation of exactpathgrad: 4.618921074		4.267426902
# execution of exactpathgrad: 	1.386364365		1.920655101

#noise, nc = [0.25, 0.25, 0.25, 0], [20,20,20, 20], ratio 1/4
# compilation of samplinggrad: 1.037791833
# execution of samplinggrad: 1.033145842
# compilation of exactpathgrad: 4.89926781
# execution of exactpathgrad: 4.620349956
#16 threads, noise, nc = [0.25, 0.25, 0.25, 0], [10,10,10, 10], ratio 1/4





#1 thread