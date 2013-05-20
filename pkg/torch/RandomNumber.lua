-- Wrap the C functions to return/accept a table
local _setRNGState = torch._setRNGState
torch._setRNGState = nil
function torch.setRNGState(state)
    return _setRNGState(unpack(state))
end

local _getRNGState = torch._getRNGState
torch._getRNGState = nil
function torch.getRNGState()
    return {_getRNGState()}
end
