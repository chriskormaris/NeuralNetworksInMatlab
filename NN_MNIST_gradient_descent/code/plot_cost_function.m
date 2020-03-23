function fig = plot_cost_function( estimate_vector )

iterations = length(estimate_vector);
x = 1:iterations;
fig = plot(x, estimate_vector, '-');
xlabel('iterations / epochs')
ylabel('Cost function')

end
