using DrWatson 
# quickactivate(@__DIR__)
using Flux, Flux.Data.MNIST, SumDenseProduct, IterTools, StatsBase, Distributions, Unitary, Serialization
using MLDataPattern, Images, Random
using Base.Iterators: partition
using Flux: throttle, train!, Params
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
	("--layers"; arg_type = Int; default = 2);
	("--components"; arg_type = Int; default = 10);
	("--noise"; arg_type = Float64; default = 0.5);
	("--repetition"; arg_type = Int; default = 1);
	("--iterations"; arg_type = Int; default = 20000);
	("--maxpath"; arg_type = Int; default = 100);
	("--batchsize"; arg_type = Int; default = 100);
	("--ratio"; arg_type = Int; default = 4);
	("--digit"; arg_type = Int; default = -1);
	("--gradmethod"; arg_type = String; default = "sampling");

end

function preparedataset(ratio, digit)
	imgs = MNIST.images();
	if digit > -1
		imgs = imgs[MNIST.labels() .== digit]
	end
	if ratio > 1
		imgs = map(x -> imresize(x, ratio = 1/ratio), imgs);
	end
	Float32.(float(hcat(vec.(imgs)...)))
end

noisedistribution(α, d, l) = vcat([round(Int, d*α^i) for i in 1:l-1], 0)

function experiment(s)
	X = preparedataset(s[:ratio], s[:digit])
	Random.seed!(Random.GLOBAL_RNG, s[:repetition])
	xtrn, xtst = splitobs(X, 0.8)
	d = size(xtrn,1)
	oprefix = datadir("..","mnist",savename(s))

	nc = fill(s[:components], s[:layers])
	model = buildmixture(d, nc, fill(identity, length(nc)), noisedistribution(d, s[:noise], s[:layers]));


	history = fit!(model, xtrn, s[:batchsize], s[:iterations], s[:maxpath], xval = xtst, gradmethod = Symbol(s[:gradmethod]));
	serialize(oprefix*"_model.jls",model)
	serialize(oprefix*"_history.jls",history)
	o = SumDenseProduct.batchlogpdf(model, xtst, s[:batchsize])
	serialize(oprefix*"_logpdf.jls",o)
end


quickactivate(@__DIR__)
settings = parse_args(ARGS, s; as_symbols=true)
experiment(settings)