if (!"sp" %in% installed.packages()) install.packages("sp")
library(sp)
if (!"RColorBrewer" %in% installed.packages()) install.packages("RColorBrewer")
library(RColorBrewer)
if (!"ggplot2" %in% installed.packages()) install.packages("ggplot2")
library(ggplot2)

# *** Se carga el fichero con los datos de España
spain <- readRDS("ESP_adm4.rds")
# *** Se carga los datos de tenerife, con los municipios
canary <- spain[spain$NAME_2 == c("Santa Cruz de Tenerife"),]
#Gomera
tfe <- canary[canary$NAME_4 != c("Agulo"),]
tfe <- tfe[tfe$NAME_4 != c("Alajeró"),]
tfe <- tfe[tfe$NAME_4 != c("Hermigua"),]
tfe <- tfe[tfe$NAME_4 != c("San Sebastián de la Gomera"),]
tfe <- tfe[tfe$NAME_4 != c("Vallehermoso"),]
tfe <- tfe[tfe$NAME_4 != c("Valle Gran Rey"),]
#La Palma
tfe <- tfe[tfe$NAME_4 != c("Barlovento"),]
tfe <- tfe[tfe$NAME_4 != c("Breña Alta"),]
tfe <- tfe[tfe$NAME_4 != c("Breña Baja"),]
tfe <- tfe[tfe$NAME_4 != c("Fuencaliente de la Palma"),]
tfe <- tfe[tfe$NAME_4 != c("Garafía"),]
tfe <- tfe[tfe$NAME_4 != c("Los Llanos de Aridane"),]
tfe <- tfe[tfe$NAME_4 != c("El Paso"),]
tfe <- tfe[tfe$NAME_4 != c("Puntagorda"),]
tfe <- tfe[tfe$NAME_4 != c("Puntallana"),]
tfe <- tfe[tfe$NAME_4 != c("San Andrés y Sauces"),]
tfe <- tfe[tfe$NAME_4 != c("Santa Cruz de la Palma"),]
tfe <- tfe[tfe$NAME_4 != c("Tazacorte"),]
tfe <- tfe[tfe$NAME_4 != c("Tijarafe"),]
tfe <- tfe[tfe$NAME_4 != c("Villa de Mazo"),]
#Hierro
tfe <- tfe[tfe$NAME_4 != c("Frontera"),]
tfe <- tfe[tfe$NAME_4 != c("Valverde"),]

##Cargar datos
estaciones <- read.csv("Estaciones.csv", header = TRUE, sep = ";", stringsAsFactors=FALSE)

tfe$Municipality <- tfe$NAME_4
#tfe@data <- merge(tfe@data, densityCA, by="Municipality")
#tfe$Municipality <- NULL  # Remove the temporary variable
plot(tfe,
     main = "",
     xlab = "Longitude",
     ylab = "Longitude")
axis(1)
axis(2, las = "2")
points(estaciones$Lon, estaciones$Lat, col = "red", cex = 1, pch = 17)
text(estaciones$Lon, estaciones$Lat, estaciones$ID, pos = 4, cex = 0.8)
