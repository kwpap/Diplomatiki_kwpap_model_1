# install.packages("ggplot2")
# install.packages("DBI")
# install.packages("RMariaDB")
# install.packages("RMySQL")
# install.packages("broom")
# install.packages("ggpubr")

library("DBI")
library("RMariaDB")
library("RMySQL")
library("ggplot2")
library("broom")
library("ggpubr")


# DEFINITIONS FOR DATABASE
db <- "eu_ets"           # name of database
use <- "root"           # user name
passwor <- ""     # password
hos <- "localhost"       # host name

# DEFINITIONS FOR MODEL

need_diagrams_of_random_data <- FALSE
# TRUE if you want to generate diagrams of imported data for easy debugging
countrs_id_for_diagrams <- 10
# number that indicated which country's data you want to generate diagrams for
first_year_of_calculation <- 1990
last_year_of_calculation <- 2018
disp_lin_regration_diagrams <- FALSE
# TRUE if you want to generate diagrams of linear regression
prediction_year <- 2018

# DEFINITIONS OF FUNCTIONS
year <- function(x) {
  return(x - 1960)
}
normalize_dictionary <- function(dictionary) {
  return(dictionary / dictionary[1])
} # thelei allagi
find_the_row_of_a_country <- function(country_name, data_frame) {
  for (i in 1:nrow(data_frame)) {
    if (data_frame[, 1][i] == country_name) {
      return(i)
    }
  }
}

my_linear_regration <- function(xx, yy, start, end, disp_diagrams, country_name) {
  #demo data for debugging
  # xx <- GDP_df
  # yy <- CO2_bau_df
  # start <- 2000
  # end <- 2010
  # disp_diagrams <- TRUE
  # country_name <- "Greece"
  #end of demo data for debugging

  row_x = find_the_row_of_a_country(country_name, xx)
  row_y = find_the_row_of_a_country(country_name, yy)
  if (row_x == 0 || row_y == 0) {
    return(0)
  }
  start <- start - 1960 + 2
  end <- end - 1960 + 2
  if (start < 1) {
    start <- 1
  }
  if (end < 1) {
    end <- 1
  }
  if (start > 60) {
    start <- 60
  }
  if (end > 60) {
    end <- 60
  }
  while (start <= end) {
    if (is.na(xx[row_x, start]) || is.na(yy[row_y, start]) || xx[row_x, start] == 0 || yy[row_y, start] == 0) {
      start <- start + 1
    } else {
      break
    }
  }
  while ( start <= end) {
    if (is.na(xx[row_x, start]) || is.na(yy[row_y, start]) || xx[row_x, end] == 0 || yy[row_y, end] == 0) {
      end <- end - 1
    } else {
      break
    }
  }
  if (start >= end) {
    return(0)
  }

  x_data <- t(xx[row_x, start:end])
  y_data <- t(yy[row_y, start:end])
  colnames(x_data) <- c("x")
  colnames(y_data) <- c("y")
  lm <- lm(y_data ~ x_data)
  # display the linear regression diagram
  if (disp_diagrams) {
#create regression plot with customized style
    data <- cbind(y_data, x_data)
    data <- data.frame(data)
    ggplot(data, aes(x <- x_data, y <= y_data)) +
      geom_point() +
      geom_smooth(method='lm', se=FALSE, color='turquoise4') +
      theme_minimal() +
      labs(x='X Values', y='Y Values', title='Linear Regression Plot') +
      theme(plot.title = element_text(hjust=0.5, size=20, face='bold')) 

    plot(lm)
    title(paste("Linear regression diagram for ", country_name))
    # legend(legend = c("Population", "Linear regression"))
    dev.off()
  }
    return(lm)
}


# IMPORT DATA

# import countries from databse
kanali <- dbConnect(RMariaDB::MariaDB(),
                    user = use,
                    password = passwor,
                    dbname = db,
                    host = hos)
qurry_countries <- "SELECT name, abbr2L, eu_abbr2L from countries where EU =1"
res <- dbSendQuery(kanali, qurry_countries) # send query to database
countries <- dbFetch(res, n = -1) # fetch all data from querry
dbClearResult(res) # clear result
country_names <- countries[, 1]
country_abbr2L <- countries[, 2] # country's abbreviation  # nolint
country_eu_abbr2L <- countries[, 3] # country's EU abbreviation  # nolint

# import verified emisions from databse for each country from 2005 to 2020

for (i in 1:length(country_names)) {
  df_temp <- data.frame()
  rv <- vector()
  for (j in 2005:2021){
    querr <- paste(
      "SELECT SUM(verified) FROM `eutl_compliance` WHERE country = '",
    country_eu_abbr2L[i], "' AND etos ='", j, "'", sep = "", collapse = NULL)
    res <- dbSendQuery(kanali, querr) # send query to database
    verified <- dbFetch(res, n = -1) # fetch all data from querry
    dbClearResult(res) # clear result
    rv <- c(rv, verified[1,1])
  }
  df_temp <- data.frame(rv)
  colnames(df_temp) <- c(country_names[i])
  # print(paste("Autes einai oi malakies tis", country_names[i], rv))
  if (i == 1) {
    df <- df_temp
  } else {
    df <- cbind(df, df_temp)
  }
}
verified_df <- df


###############################################################################

# Import polutation data from csv
# Headers are on the 4rth row, thus we read them seperately
headers <- read.csv(file = "API_SP.POP.TOTL_DS2_en_csv_v2_3731322.csv",
                    skip = 4,
                    header = FALSE,
                    nrows = 1,
                    as.is = TRUE)
df <- read.csv(file = "API_SP.POP.TOTL_DS2_en_csv_v2_3731322.csv",
                    skip = 4, header = FALSE)
colnames(df) <- headers
# Άρα, έχει διαβαστεί το αρχείο και έχει στην θέση 1 το όνομα της κάθε χώρας
# στην θέση 5 έχει τον πληθυσμό το 1960 και στην θέση 65
pop_df <- subset(df, select = -c(2, 3, 4))
#Εδώ διαγράφουμε τις 3 περιττές στήλες.

#Import CO2 BAU data from csv ομοίως.
# Headers are on the 4rth row, thus we read them seperately
headers <- read.csv(file = "API_EN.ATM.CO2E.KT_DS2_en_csv_v2_3830791.csv",
                    skip = 4,
                    header = FALSE,
                    nrows = 1,
                    as.is = TRUE)
df <- read.csv(file = "API_EN.ATM.CO2E.KT_DS2_en_csv_v2_3830791.csv",
                    skip = 4, header = FALSE)
colnames(df) <- headers
CO2_bau_df <- subset(df, select = -c(2, 3, 4)) # nolint

#Import GDP emissions data from csv ομοίως.
# Headers are on the 4rth row, thus we read them seperately
headers <- read.csv(file = "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_3840536.csv",
                    skip = 4,
                    header = FALSE,
                    nrows = 1,
                    as.is = TRUE)
df <- read.csv(file = "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_3840536.csv",
                    skip = 4, header = FALSE)
colnames(df) <- headers
GDP_df <- subset(df, select = -c(2, 3, 4)) # nolint

###############################################################################

#perform the linear regression for each country.

for (i in 1:length(country_names)) {
  if (country_names[i] == "Slovakia") {
    lm <- linear_regression(GDP_df, CO2_bau_df, 1960, 2020, TRUE, "Slovak Republic")
    continue
  }
  lm <- my_linear_regration(GDP_df, CO2_bau_df, 1960, 2020, TRUE, country_names[i])
  # save the linear regression model
  # save(lm, file = paste("lm_",
  # country_names[i], ".RData", sep = ""))
}


dbDisconnect(kanali) # close connection to database

