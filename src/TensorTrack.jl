module TensorTrack

using TensorOperations
import TensorOperations: add!, trace!, contract!,
    IndexTuple, similar_from_indices, cached_similar_from_indices

using Strided
import Strided: StridedView, UnsafeStridedView

using TupleTools

include("backwards.jl")

include("tracker.jl")

include("zygote.jl")

end
