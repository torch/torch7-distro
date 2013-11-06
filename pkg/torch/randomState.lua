function torch.getRNGState()
    return torch.getMTState()
end

function torch.setRNGState(state)
    return torch.setMTState(state)
end
