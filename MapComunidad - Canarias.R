
# Activamos/Instalamos las librer?as
if (!"sp" %in% installed.packages()) install.packages("sp")
library(sp)
if (!"RColorBrewer" %in% installed.packages()) install.packages("RColorBrewer")
library(RColorBrewer)
#
# Seleccionamos los datos de Canarias
spain <- readRDS("ESP_adm4.rds")


#####################################################################
unique(spain$NAME_1)
AllCanary <- spain[spain$NAME_1 == c("Islas Canarias"),]
plot(AllCanary, axes =TRUE, sub="Comunidad de Canarias")
#title("Comunidad de Canarias")
######################################################################

canary <- spain[spain$NAME_2 == c("Santa Cruz de Tenerife"),]
#
# Seleccionamos los datos de la isla de Tenerife
# (Quitamos los municipios de las otras islas)
# Gomera
tfe <- canary[canary$NAME_4 != c("Agulo"),]
tfe <- tfe[tfe$NAME_4 != c("Alajer?"),]
tfe <- tfe[tfe$NAME_4 != c("Hermigua"),]
tfe <- tfe[tfe$NAME_4 != c("San Sebasti?n de la Gomera"),]
tfe <- tfe[tfe$NAME_4 != c("Vallehermoso"),]
tfe <- tfe[tfe$NAME_4 != c("Valle Gran Rey"),]
# La Palma
tfe <- tfe[tfe$NAME_4 != c("Barlovento"),]
tfe <- tfe[tfe$NAME_4 != c("Bre?a Alta"),]
tfe <- tfe[tfe$NAME_4 != c("Bre?a Baja"),]
tfe <- tfe[tfe$NAME_4 != c("Fuencaliente de la Palma"),]
tfe <- tfe[tfe$NAME_4 != c("Garaf?a"),]
tfe <- tfe[tfe$NAME_4 != c("Los Llanos de Aridane"),]
tfe <- tfe[tfe$NAME_4 != c("El Paso"),]
tfe <- tfe[tfe$NAME_4 != c("Puntagorda"),]
tfe <- tfe[tfe$NAME_4 != c("Puntallana"),]
tfe <- tfe[tfe$NAME_4 != c("San Andr?s y Sauces"),]
tfe <- tfe[tfe$NAME_4 != c("Santa Cruz de la Palma"),]
tfe <- tfe[tfe$NAME_4 != c("Tazacorte"),]
tfe <- tfe[tfe$NAME_4 != c("Tijarafe"),]
tfe <- tfe[tfe$NAME_4 != c("Villa de Mazo"),]
# Hierro
tfe <- tfe[tfe$NAME_4 != c("Frontera"),]
tfe <- tfe[tfe$NAME_4 != c("Valverde"),]
#
# Colores
mapcolors <- read.csv("DataRSU.csv", header=T, sep=";")
tfe2 <- merge(tfe, mapcolors, by.x="NAME_4", by.y="Municipality", all = TRUE)
#
# Dibujamos
# Dividimos la ventana de dibujo en 4...
par(mfrow=c(2,2))
# En una de las zonas...
plot(tfe)
# En otra de las zonas...
# (Por cuesti?n del tama?o de la ventana gr?fica, priemro toda la isla
#  y despu?s en colores, municipio por municipio)
plot(tfe)
for(i in 1:31) {plot(tfe[i,1], col=i, add=T)}
plot(tfe2)
for(i in 1:31) {plot(tfe2[i,tfe2$data1], col=i, add=T)}
#
# Se salva a un fichero jpg (raster), con fondo color aguamarina
jpeg("Tenerife.jpg")
plot(tfe, bg="aquamarine3")
for(i in 1:31) {plot(tfe[i,1], col=i, add=T)}
dev.off()
#
# Se salva a un fichero svg (vectorial)
svg("Tenerife.svg")
plot(tfe)
for(i in 1:31) {plot(tfe[i,1], col=i, add=T)}
dev.off()

