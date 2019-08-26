# rsync -avz aws-ecs:/mnt/output/results/datasets ~/Work/Julia/results/
using DrWatson, BSON, DataFrames, PrettyTables, Crayons, Statistics

priorart = DataFrame(
dataset = ["breast-cancer-wisconsin", "cardiotocography", "magic-telescope", "pendigits", "pima-indians", "wall-following-robot", "waveform-1", "waveform-2", "yeast"],
iforest = [0.86, 0.63, 0.81, 0.58, 0.87, 0.52, 0.54, 0.53, 0.66],
knn =     [0.86, 0.52, 0.9 , 0.52, 0.88, 0.44, 0.47, 0.48, 0.63],
svae_m2 = [0.92, 0.88, 0.92, 0.69, 0.94, 0.62, 0.82, 0.70, 0.83],
vae_m1 =  [0.77, 0.70, 0.85, 0.57, 0.85, 0.50, 0.49, 0.51, 0.69])

const resultsdir = filter(isdir,["/Users/tpevny/Work/Julia/results/datasets","/mnt/output/results/datasets","/opt/output/results/datasets"])[1];

function collectdir(problem)
	# res = collect_results(joinpath(resultsdir,problem,"sumdense"); white_list = [:stats], valid_filetypes = ["_stats.bson"])[:,1]
	files = filter(s ->endswith(s,"_stats.bson"),readdir(joinpath(resultsdir,problem,"sumdense")))
	res = map(s -> deserialize(joinpath(resultsdir, problem, "sumdense", s)), files)
	res = filter(s -> typeof(s) <: NamedTuple, res)

	# we need to do a manual reduction with missing values
	df = DataFrame()
	# for k in mapreduce(keys, (u,v) -> union(u, v), res)
	for k in keys(res[1])
		df[!, k] = map(r -> haskey(r, k) ? r[k] : missing, res)
	end
	df[!, :problem] .= problem
	df
end

function collectresults()
	# problems = readdir(resultsdir);
	# problems = filter(s -> isdir(joinpath(resultsdir, s,"sumdense")), problems); 
	# problems = filter(s -> !isempty(readdir((joinpath(resultsdir, s,"sumdense")))), problems); 
	problems = priorart[:dataset]
	df = map(collectdir, problems)
	reduce(vcat, df)
end


function show()
	df = collectresults()
	dflkl = by(df, [:dataset, :repetition], dff -> DataFrame(SumDenseLkl = dff[argmax(dff[:test_lkl]),:test_auc], n = length(dff)))
	dflkl = by(dflkl, :dataset, dff -> DataFrame(SumDenseLkl = mean(dff[:SumDenseLkl])))
	dfauc = by(df, [:dataset, :repetition], dff -> DataFrame(SumDenseAUC = dff[argmax(dff[:train_auc]),:test_auc], n = length(dff)))
	dfauc = by(dfauc, :dataset, dff -> DataFrame(SumDenseAUC = mean(dff[:SumDenseAUC])))
	dff = join(dflkl,dfauc, on = :dataset)
	dff = join(dff,priorart, on = :dataset)
	h1 = Highlighter(
		f = (data, i, j) -> data[i,j] == maximum(dff[i, 2:end]),
		crayon = crayon"yellow bold")
	pretty_table(dff;  formatter=ft_round(2), highlighters = h1)
	dff = dff[:,[:dataset, :SumDenseAUC,:iforest,:knn,:vae_m1]]
	pretty_table(dff;  formatter=ft_round(2), highlighters = h1)
end
