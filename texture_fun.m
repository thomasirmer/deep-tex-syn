function [z, dzdx] = texture_fun(x, net, objectiveString, opts)
if strcmp(net.device, 'gpu')
  x = gpuArray(x);
end

inputs = {'input', reshape(x, opts.imageSize)};
net.eval(inputs, objectiveString);
z = 0;
for i = 1:length(objectiveString)/2,
    z_ = net.vars(net.getVarIndex(objectiveString(2*i-1))).value;
    w_ = objectiveString{2*i};
    z = z + z_*w_;
end
dzdx = net.vars(net.getVarIndex('input')).der;

% Gradient of TV norm
[tvz, tvdzdx] = tv(reshape(x, opts.imageSize), opts.TVbeta);
z = z + opts.lambdaTV/2*tvz;
dzdx = dzdx + opts.lambdaTV/2*tvdzdx;

dzdx = dzdx(:);

if strcmp(net.device, 'gpu')
  z = gather(z); 
  dzdx = gather(dzdx);
end



function [e, dx] = tv(x,beta)
d1 = x(:,[2:end end],:,:) - x ;
d2 = x([2:end end],:,:,:) - x ;
v = sqrt(d1.*d1 + d2.*d2).^beta ;
e = sum(sum(sum(sum(v)))) ;
if nargout > 1
  d1_ = (max(v, 1e-5).^(2*(beta/2-1)/beta)) .* d1;
  d2_ = (max(v, 1e-5).^(2*(beta/2-1)/beta)) .* d2;
  d11 = d1_(:,[1 1:end-1],:,:) - d1_ ;
  d22 = d2_([1 1:end-1],:,:,:) - d2_ ;
  d11(:,1,:,:) = - d1_(:,1,:,:) ;
  d22(1,:,:,:) = - d2_(1,:,:,:) ;
  dx = beta*(d11 + d22);
end

