
library(circular)

# Read the CSV file with one column and no header
CurrentFixationAngles <- read.csv('./CCREG/current_fixation_angle.csv', header = FALSE, col.names = c("Values"))$Values
NextFixationAngles <- read.csv('./CCREG/next_fixation_angle.csv', header = FALSE, col.names = c("Values"))$Values
RandomAnglesModel4 <- read.csv('./CCREG/random_angle_model4.csv', header = FALSE, col.names = c("Values"))$Values
RandomAnglesModel5 <- read.csv('./CCREG/random_angle_model5.csv', header = FALSE, col.names = c("Values"))$Values

# Fit a circular-circular regression model with the current and next fixation angles
order <- 1
circ.lm <- lm.circular(NextFixationAngles, CurrentFixationAngles, order=order, type="c-c")
while (any(circ.lm$p.values <= 0.05) && order <= 5) {
  order <- order + 1
  circ.lm <- lm.circular(NextFixationAngles, CurrentFixationAngles, order=order, type="c-c")
}
print("Order of the model:")
print(order)
# Fit a circular-circular regression model with the current fixation angles and random angles from model 4
order <- 1
circ.lm4 <- lm.circular(RandomAnglesModel4, CurrentFixationAngles , order=order, type="c-c")
while (any(circ.lm4$p.values <= 0.05) && order <= 5) {
  order <- order + 1
  circ.lm4 <- lm.circular(RandomAnglesModel4, CurrentFixationAngles, order=order, type="c-c")
}

print("Order of the model R4:")
print(order)
# Fit a circular-circular regression model with the current fixation angles and random angles from model 5
order <- 1
circ.lm5 <- lm.circular(RandomAnglesModel5, CurrentFixationAngles , order=order, type="c-c")
while (any(circ.lm5$p.values <= 0.05) && order <= 5) {
  order <- order + 1
  circ.lm5 <- lm.circular(RandomAnglesModel5, CurrentFixationAngles, order=order, type="c-c")
}

print("Order of the model R5:")
print(order)

# Print the results
print(circ.lm$p.values)
print(circ.lm4$p.values)
print(circ.lm5$p.values)

#Save the p.values to a file
write.csv(as.data.frame(circ.lm$p.values),  file = "./CCREG/p_values.csv")
write.csv(as.data.frame(circ.lm4$p.values), file = "./CCREG/p_valuesR4.csv")
write.csv(as.data.frame(circ.lm5$p.values), file = "./CCREG/p_valuesR5.csv")

#Save the coefficients to a file
write.csv(as.data.frame(circ.lm$coefficients), file = "./CCREG/fit.csv")
write.csv(as.data.frame(circ.lm4$coefficients), file = "./CCREG/fitR4.csv")
write.csv(as.data.frame(circ.lm5$coefficients), file = "./CCREG/fitR5.csv")

#Create a sequence for ordering the fitted values
#x <- seq(0, 2 * pi, length.out = length(CurrentFixationAngles))
x <- CurrentFixationAngles
# Save the fitted values to files in an ordered manner
data_to_save <- data.frame(Ordered_X = x[order(x)], Fitted_Values = circ.lm$fitted[order(x)])
write.table(data_to_save, file = "./CCREG/fitted_values_ordered.txt",  sep = "\t", row.names = FALSE)

data_to_save4 <- data.frame(Ordered_X = x[order(x)], Fitted_Values = circ.lm4$fitted[order(x)])
write.table(data_to_save4, file = "./CCREG/fitted_values_orderedR4.txt", sep = "\t", row.names = FALSE)

data_to_save5 <- data.frame(Ordered_X = x[order(x)], Fitted_Values = circ.lm5$fitted[order(x)])
write.table(data_to_save5, file = "./CCREG/fitted_values_orderedR5.txt", sep = "\t", row.names = FALSE)