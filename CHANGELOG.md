# Changelog - Sistema de Planificación Urbana

## Versión 2.0 - Sistema con Reordenamiento Dinámico (2025-01-18)

### ✨ Nuevas Características Principales

#### 1. **Sistema de Reordenamiento Dinámico**
- ✅ Implementación de intercambio de posiciones entre hogares y servicios
- ✅ Mantenimiento constante del número de hogares durante la optimización
- ✅ Pool unificado de ubicaciones disponibles para reasignación
- ✅ Optimización que considera tanto hogares como servicios como variables

#### 2. **Problema de Optimización Mejorado**
- ✅ Nueva clase `ReorderingProblem` que permite reasignación completa
- ✅ Variables binarias: cada ubicación puede ser hogar (0) o servicio (1)
- ✅ Restricción dura: mantener exactamente N hogares
- ✅ Función objetivo dual: maximizar cobertura + mantener balance

#### 3. **Optimización Iterativa por Categorías**
- ✅ Proceso iterativo que optimiza una categoría a la vez
- ✅ Actualización incremental del estado urbano
- ✅ Historial completo de métricas por iteración
- ✅ Convergencia gradual hacia configuración óptima

#### 4. **Visualización Comparativa Mejorada**
- ✅ Mapa interactivo con estado inicial vs optimizado
- ✅ Capas superpuestas para comparación directa
- ✅ Leyenda detallada con explicación de símbolos
- ✅ Control de capas para análisis selectivo

### 📊 Mejoras en Métricas y Evaluación

- ✅ Cálculo de cobertura para múltiples categorías simultáneamente
- ✅ Métricas de balance en la distribución de servicios
- ✅ Análisis de impacto cruzado entre categorías
- ✅ Exportación de historial completo de evolución

### 🔧 Mejoras Técnicas

- ✅ Optimización de cálculo de Dijkstra multi-fuente
- ✅ Manejo eficiente de grandes conjuntos de ubicaciones
- ✅ Validación de restricciones en tiempo de ejecución
- ✅ Selección automática de mejor solución del frente de Pareto

### 📦 Archivos y Documentación

- ✅ README.md completo con guía detallada
- ✅ INICIO_RAPIDO.md para comenzar rápidamente
- ✅ ejemplo_rapido.py para pruebas inmediatas
- ✅ configuracion_avanzada.py para personalización
- ✅ verificar_instalacion.py para diagnóstico
- ✅ requirements.txt con todas las dependencias

### 🐛 Correcciones

- ✅ Corrección de error en cálculo de cobertura con servicios vacíos
- ✅ Manejo robusto de casos sin datos en OSM
- ✅ Validación de parámetros de entrada
- ✅ Mejor manejo de errores de geocodificación

---

## Versión 1.0 - Sistema Base (Versión Original)

### Características Base

#### 1. **Carga de Datos Geoespaciales**
- Descarga de red peatonal desde OpenStreetMap
- Extracción de puntos de interés (servicios)
- Muestreo de ubicaciones residenciales
- Clasificación por categorías (salud, educación, áreas verdes, trabajo)

#### 2. **Evaluación de Accesibilidad**
- Cálculo de tiempos mínimos usando algoritmo de Dijkstra
- Evaluación de cobertura por categoría
- Métricas de accesibilidad basadas en red real
- Concepto de Ciudad de 15 Minutos

#### 3. **Optimización con NSGA-II**
- Selección binaria de nuevos sitios de servicios
- Optimización multi-objetivo (cobertura vs costo)
- Generación de frente de Pareto
- Propuesta de ubicaciones óptimas para nuevos servicios

#### 4. **Visualización**
- Mapas interactivos con Folium
- Exportación de resultados en GeoJSON
- Visualización de hogares cubiertos/no cubiertos
- Diferenciación por categorías de servicios

#### 5. **Infraestructura de Ciclovías**
- Soporte para análisis de ciclovías
- Cálculo de tiempos en bicicleta
- Marcado de vías con infraestructura ciclista

---

## Comparación de Versiones

| Característica | v1.0 | v2.0 |
|----------------|------|------|
| Optimización de servicios | ✅ Solo añadir nuevos | ✅ Reubicar existentes |
| Intercambio hogares-servicios | ❌ No | ✅ Sí |
| Número de hogares constante | ❌ No garantizado | ✅ Garantizado |
| Optimización iterativa | ❌ Una sola vez | ✅ Múltiples iteraciones |
| Visualización comparativa | ❌ Solo estado final | ✅ Inicial vs Final |
| Balance de servicios | ❌ No considerado | ✅ Optimizado |
| Documentación | ⚠️ Básica | ✅ Completa |
| Scripts de ejemplo | ❌ No | ✅ Sí |

---

## Roadmap Futuro

### Versión 2.1 (Planificado)

- [ ] Integración de restricciones de zonificación
- [ ] Consideración de costos económicos reales
- [ ] Análisis de sensibilidad de parámetros
- [ ] Generación automática de reportes PDF
- [ ] Soporte para múltiples escenarios paralelos

### Versión 2.2 (En consideración)

- [ ] Interfaz gráfica de usuario (GUI)
- [ ] Integración con bases de datos municipales
- [ ] Análisis temporal (cambios a lo largo del tiempo)
- [ ] Exportación a formatos GIS estándar (KML, GPKG)
- [ ] API REST para integración con otros sistemas

### Versión 3.0 (Futuro)

- [ ] Optimización con Deep Learning
- [ ] Predicción de demanda futura
- [ ] Análisis de tráfico vehicular
- [ ] Integración con transporte público
- [ ] Modelo 3D de la ciudad

---

## Notas de Migración

### De v1.0 a v2.0

**Cambios en la API:**

1. **Función principal**
   - Antes: `run_nsga2_siting()`
   - Ahora: `run_reordering_optimization()` + `iterative_reordering()`

2. **Parámetros**
   - Nuevo: `--iterations` para número de iteraciones
   - Nuevo: `--categories` para seleccionar categorías
   - Modificado: Comportamiento de optimización es diferente

3. **Archivos de salida**
   - Antes: `nsga2_new_{categoria}.geojson`
   - Ahora: `homes_optimized.geojson`, `services_{categoria}_optimized.geojson`
   - Nuevo: `comparison_map.html` con visualización mejorada
   - Nuevo: `optimization_history.csv` con evolución completa

**Compatibilidad:**

- ✅ Los archivos GeoJSON de v1.0 pueden ser visualizados en v2.0
- ✅ El formato de datos de entrada es compatible
- ⚠️ Los scripts que llaman directamente a funciones internas requieren actualización

---

## Créditos y Contribuciones

### Desarrolladores
- **Versión 1.0**: Carolina (código base)
- **Versión 2.0**: Mejoras implementadas por Claude AI

### Basado en Investigación
- NSGA-II: Deb et al. (2002)
- OSMnx: Boeing (2017)
- Ciudad de 15 Minutos: Moreno et al. (2021)

### Herramientas y Librerías
- Python 3.8+
- OSMnx para datos geoespaciales
- NetworkX para grafos
- GeoPandas para GIS
- pymoo para optimización
- Folium para visualización

---

## Licencia

MIT License - Libre uso para fines académicos y comerciales

---

**Última actualización**: 18 de Enero, 2025
