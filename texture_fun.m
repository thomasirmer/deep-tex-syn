function [z, dzdx] = texture_fun(x, net, objectiveString, opts)
inputs = {'input', reshape(x, net.meta.normalization.imageSize)};
net.eval(inputs, objectiveString);
z = 0;
for i = 1:length(objectiveString)/2,
    z_ = net.vars(net.getVarIndex(objectiveString(2*i-1))).value;
    w_ = objectiveString{2*i};
    z = z + z_*w_;
end
dzdx = net.vars(net.getVarIndex('input')).der;
dzdx = dzdx(:);