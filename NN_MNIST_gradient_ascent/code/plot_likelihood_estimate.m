function fig = plot_likelihood_estimate( estimate_vector )

iterations = length(estimate_vector);
x = 1:iterations;
fig = plot(x, estimate_vector, '-');
xlabel('iterations / epochs')
ylabel('Likelihood estimate')

end
