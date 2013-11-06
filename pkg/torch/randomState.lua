function torch.getRNGState()
    return {
        mt = torch.getMTState(),
        normal = torch.getNormalState()
    }
end

function torch.setRNGState(state)
    torch.setMTState(state.mt)
    torch.setNormalState(state.normal)
end
