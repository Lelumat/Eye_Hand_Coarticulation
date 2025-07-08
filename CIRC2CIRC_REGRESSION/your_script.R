# Load circular package
suppressMessages(suppressWarnings({
    library(circular)
    }))

# Import data from CSV files
independent_angle <- read.csv("./CCREG/independent_angle_train_fold.csv", header = FALSE, col.names = c("Values"))$Values
dependent_angle <- read.csv("./CCREG/dependent_angle_train_fold.csv", header = FALSE, col.names = c("Values"))$Values

# Convert data to circular objects
independent_angle <- circular(independent_angle, units = "radians")
dependent_angle <- circular(dependent_angle, units = "radians")

# Initialize variables
order <- 1
circ.lm <- lm.circular(dependent_angle, independent_angle, order = order)

# Loop until both p-values are above 0.05 or order goes beyond 5
while (any(circ.lm$p.values <= 0.05) && order <= 5) {
  order <- order + 1
  circ.lm <- lm.circular(dependent_angle, independent_angle, order = order)
}
#Save the coefficients to a file
write.csv(as.data.frame(circ.lm$coefficients), file = "./CCREG/fit_fold.csv")

#x <- independent_angle
# Save the fitted values to files in an ordered manner
#data_to_save <- data.frame(Ordered_X = x[order(x)], Fitted_Values = circ.lm$fitted[order(x)])
#write.table(data_to_save, file = "./CCREG/fitted_values_ordered_fold.txt",  sep = "\t", row.names = FALSE)
