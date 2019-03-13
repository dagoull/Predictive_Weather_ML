library(stationaRy)
library(dplyr)

# https://github.com/rich-iannone/stationaRy
##MyStation <-  get_isd_stations(startyear = 2014, endyear = 2015)

# Get a tibble with all stations in Spain
stations_spain_canary <- get_isd_stations() %>%  filter(country == "SP", gmt_offset == "0")

#met_data <- get_isd_station_data(station_id = "600180-99999", startyear = 2010, endyear = 2011)
#met_data_SANTACRUZ <- get_isd_station_data(station_id = "600200-99999", startyear = 2010, endyear = 2014, select_additional_data = "AO1")
#met_data_TFN <- get_isd_station_data(station_id = "600150-99999", startyear = 2015, endyear = 2015)


met_data_TFN <- get_isd_station_data(station_id = "600200-99999", startyear = 2015, endyear = 2015, select_additional_data = "AA1")

library(magrittr)
#Select stations using a bounding box and map the stations 

get_isd_stations(lower_lat = 28.000,
                 upper_lat = 28.500,
                 lower_lon = -16.500,
                 upper_lon = -16.000) %>%
map_isd_stations()

#28.696886, -16.893414
#28.034992, -16.060189

#get_isd_stations(lower_lat = 49.000,
#upper_lat = 49.500,
#lower_lon = -123.500,
#upper_lon = -123.000) %>%