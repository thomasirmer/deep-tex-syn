classdef L2Loss < dagnn.ElementWise
  properties
    loss = 'l2'
    opts = {}
  end

  properties (Transient)
    average = 0
    numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
        
      d = inputs{1} - inputs{2};
      outputs{1} = sum(sum(sum(sum(d.*d))));
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = derOutputs{1} * 2 * (inputs{1} - inputs{2});
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = L2Loss(varargin)
      obj.load(varargin) ;
    end
  end
end
