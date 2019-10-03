

"""
	updatelatent!(m, x, bs::Int = typemax(Int))

	estimate the probability of a component in `m` using data in `x`.
	if `bs < size(x,2)`, then the update is calculated part by part to save memory
"""
function updatelatent!(m, x, bs::Int = typemax(Int))
	zerolatent!(m);
	lkl, paths = mappath(m, x)
	foreach(p -> _updatelatent!(m, p), path)
	normalizelatent!(m);
end

zerolatent!(m) = nothing
normalizelatent!(m) = nothing
_updatelatent!(m, path) = nothing
