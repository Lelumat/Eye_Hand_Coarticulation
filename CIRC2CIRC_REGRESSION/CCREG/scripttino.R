# Specify the file path where the CSV is saved
csv_file_path <- 'independent_angle.csv'

# Read the CSV file with one column and no header
x <- read.csv(csv_file_path, header = FALSE, col.names = c("Values"))$Values

# Specify the file path where the CSV is saved
csv_file_path <- 'dependent_angle.csv'

# Read the CSV file with one column and no header
y <- read.csv(csv_file_path, header = FALSE, col.names = c("Values"))$Values


# Fit a circular-circular regression model.
circ.lm <- lm.circular(y, x, order=1)
# Obtain a crude plot of the data and fitted regression line.
plot.default(x, y)
#circ.lm$fitted[circ.lm$fitted>pi] <- circ.lm$fitted[circ.lm$fitted>pi] - 2*pi 
points.default(x[order(x)], circ.lm$fitted[order(x)], type='l')



# Function to calculate cosine and sine vectors of a vector of angles in radians
calculate_cosine_sine_vectors <- function(angles_radians) {
  # Calculate cosine and sine vectors
  cosine_values <- cos(angles_radians)
  sine_values <- sin(angles_radians)
  
  # Return the result as a list
  result_list <- list(cosine = cosine_values, sine = sine_values)
  
  # Return the result
  return(result_list)
}

# Example usage:
# Replace c(0, pi/4, pi/2) with your vector of angles in radians
angles_vector <- c(0, pi/4, pi/2)
result <- calculate_cosine_sine_vectors(angles_vector)

# Access cosine and sine vectors from the result
cosine_vector <- result$cosine
sine_vector <- result$sine

# Print the result
print("Cosine Vector:")
print(cosine_vector)

print("Sine Vector:")
print(sine_vector)

