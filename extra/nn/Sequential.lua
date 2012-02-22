local Sequential, parent = torch.class('nn.Sequential', 'nn.Module')

function Sequential:__init()
   self.modules = {}
end

function Sequential:add(module)
   if #self.modules == 0 then
      self.gradInput = module.gradInput
   end
   table.insert(self.modules, module)
   self.output = module.output
   return self
end

function Sequential:size()
   return #self.modules
end

function Sequential:get(index)
   return self.modules[index]
end

function Sequential:updateOutput(input)
   local currentOutput = input
   for i=1,#self.modules do 
      currentOutput = self.modules[i]:updateOutput(currentOutput)
   end 
   self.output = currentOutput
   return currentOutput
end

function Sequential:updateGradInput(input, gradOutput)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentGradOutput = currentModule:updateGradInput(previousModule.output, currentGradOutput)
      currentModule = previousModule
   end
   currentGradOutput = currentModule:updateGradInput(input, currentGradOutput)
   self.gradInput = currentGradOutput
   return currentGradOutput
end

function Sequential:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentModule:accGradParameters(previousModule.output, currentGradOutput, scale)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end
   
   currentModule:accGradParameters(input, currentGradOutput, scale)
end

function Sequential:accUpdateGradParameters(input, gradOutput, lr)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentModule:accUpdateGradParameters(previousModule.output, currentGradOutput, lr)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end
   
   currentModule:accUpdateGradParameters(input, currentGradOutput, lr)
end

function Sequential:zeroGradParameters()
  for i=1,#self.modules do
     self.modules[i]:zeroGradParameters()
  end
end

function Sequential:updateParameters(learningRate)
   for i=1,#self.modules do
      self.modules[i]:updateParameters(learningRate)
   end
end

function Sequential:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

function Sequential:reset(stdv)
   for i=1,#self.modules do
      self.modules[i]:reset(stdv)
   end
end

function Sequential:parameters()
   local function tinsert(to, from)
      if type(from) == 'table' then
         for i=1,#from do
            tinsert(to,from[i])
         end
      else
         table.insert(to,from)
      end
   end
   local w = {}
   local gw = {}
   for i=1,#self.modules do
      local mw,mgw = self.modules[i]:parameters()
      if mw then
         tinsert(w,mw)
         tinsert(gw,mgw)
      end
   end
   return w,gw
end

function Sequential:getParameters()
	local outputParams = {}
	local outputGrad = {}
	
	for i, module in ipairs(self.modules) do
		if module:parameters() then
			local flattenedParams, flattenedGrad = module:getParameters()
			for i=1,flattenedParams:size(1) do
				table.insert(outputParams, flattenedParams[i])
				table.insert(outputGrad, flattenedGrad[i])
			end
		end
	end
	
	return torch.Tensor.new(outputParams), torch.Tensor.new(outputGrad)
end

function Sequential:fromParameters(params)
	local new = nn.Sequential()	
	local newParams = params:clone()
	local offset=1
	
	for _, module in ipairs(self.modules) do
		if module:parameters() then
			local newModule, inc = module:fromParameters(newParams, offset)
			new:add(newModule)
			offset = offset + inc
		else
			new:add(module:clone())
		end
	end
	
	return new
end

function Sequential:derivativeCheck(input, output, criterion, epsilon)
	-- first generate the gradients
	criterion:forward(self:forward(input), output)
	self:zeroGradParameters()
	self:backward(input, criterion:backward(self.output, output))	
	
	-- unroll the model's parameters and resulting gradients
	local theta, grad = self:getParameters()	
	local results = torch.Tensor(grad:size(1))
	
	-- go through each variable and perturb it +/- epsilon
	for i=1,grad:size(1) do
		-- plus
		local thetap = theta:clone()
		thetap[i] = thetap[i] + epsilon
		local seqp = self:fromParameters(thetap)
		local Jplus = criterion:forward(seqp:forward(input), output)

		-- minus
		local thetam = theta:clone()
		thetam[i] = thetam[i] - epsilon
		local seqm = self:fromParameters(thetam)
		local Jminus = criterion:forward(seqm:forward(input), output)

		-- the definition of a derivative
		local J = (Jplus - Jminus)/(2*epsilon)
		
		-- measure the difference from analytical and estimated gradients
		results[i] = math.abs(grad[i] - J)
	end
	
	return results
end

function Sequential:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'nn.Sequential'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self.modules do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self.modules do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end
