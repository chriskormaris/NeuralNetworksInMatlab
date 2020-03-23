function [ h, grad_h ] = h3( a )
%ACTIVATION_FUNCTION Summary of this function goes here
% log(1 + exp(a)) = log(exp(0) + exp(a)) = 0 + log(exp(-0) + exp(a-0))

% h
h = cos(a);

% compute the gradient of: h(a) = cos(a)
% h'(a) = -sin(a)
if nargout > 1
    grad_h = -sin(a);
end

end

