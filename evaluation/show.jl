using DrWatson, BSON, DataFrames

const resultsdir = "/Users/tpevny/Work/Julia/results/datasets/"
function collectdir(problem)
	res = collect_results(joinpath(resultsdir,problem,"sumdense"); white_list = [:stats], valid_filetypes = ["_stats.bson"])[:,1]
	res = DataFrame(res)
	res[:problem] = problem
	res
end

function collectresults()
	problems = readdir(resultsdir);
	problems = filter(s -> isdir(joinpath(resultsdir, s,"sumdense")), problems); 
	problems = filter(s -> !isempty(readdir((joinpath(resultsdir, s,"sumdense")))), problems); 
	df = map(collectdir, problems)
	reduce(vcat, df)
end


il = argmax(res[:test_lkl])
id = argmax(res[:test_dld01])
(res[il,:test_auc], res[id,:test_auc])
