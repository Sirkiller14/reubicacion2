# Sistema de Planificación Urbana con Reordenamiento Dinámico

## 📋 Descripción

Este sistema implementa un algoritmo de optimización urbana basado en NSGA-II que permite el **intercambio dinámico** entre hogares y servicios para maximizar la accesibilidad bajo el concepto de **Ciudad de 15 Minutos**.

### 🎯 Características Principales

1. **Mantenimiento de Población**: El número total de hogares se mantiene constante durante todo el proceso
2. **Intercambio Dinámico**: Los hogares y servicios pueden intercambiar posiciones para lograr una mejor distribución
3. **Optimización Iterativa**: El proceso se repite múltiples veces para diferentes categorías de servicios
4. **Multi-objetivo**: Optimiza simultáneamente la cobertura de servicios y el balance en la distribución

### 🔄 Cómo Funciona

El sistema funciona de la siguiente manera:

1. **Inicialización**: Se cargan las ubicaciones actuales de hogares y servicios desde OpenStreetMap
2. **Pool de Ubicaciones**: Se crea un conjunto de todas las ubicaciones disponibles (hogares + servicios)
3. **Optimización**: NSGA-II asigna a cada ubicación un "tipo" (hogar o servicio), manteniendo constante el número de hogares
4. **Iteración**: El proceso se repite para cada categoría de servicio (salud, educación, áreas verdes, trabajo)
5. **Resultado**: Se obtiene una nueva distribución optimizada donde hogares y servicios han intercambiado posiciones para mejorar la accesibilidad

### 📊 Objetivos de Optimización

- **Objetivo 1**: Maximizar la cobertura de hogares (% de hogares con acceso en ≤15 minutos)
- **Objetivo 2**: Mantener un balance adecuado en la proporción de servicios vs hogares
- **Restricción**: Número de hogares debe permanecer exactamente igual al inicial

---

## 🚀 Instalación

### 1. Requisitos del Sistema

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- 4GB de RAM mínimo (recomendado: 8GB)
- Conexión a Internet (para descargar datos de OpenStreetMap)

### 2. Instalar Dependencias

```bash
pip install osmnx==1.9.3 networkx==3.3 geopandas shapely rtree numpy pandas tqdm folium pymoo==0.6.1.1
```

O usando el archivo requirements.txt incluido:

```bash
pip install -r requirements.txt
```

### 3. Verificar Instalación

```python
python -c "import osmnx, pymoo, geopandas; print('Instalación exitosa')"
```

---

## 📖 Uso Básico

### Comando Simple

```bash
python ciudad_15min_reordenamiento.py --place "San Juan de Miraflores, Lima, Peru" --minutes 15 --plot
```

### Comando con Todas las Opciones

```bash
python ciudad_15min_reordenamiento.py \
    --place "San Juan de Miraflores, Lima, Peru" \
    --minutes 15 \
    --speed-kmh 4.5 \
    --max-homes 2000 \
    --iterations 3 \
    --generations 80 \
    --population 100 \
    --categories health education greens \
    --plot \
    --output-dir outputs_reordenamiento
```

---

## ⚙️ Parámetros

| Parámetro | Tipo | Por Defecto | Descripción |
|-----------|------|-------------|-------------|
| `--place` | str | **REQUERIDO** | Nombre del lugar (formato: "Distrito, Ciudad, País") |
| `--minutes` | float | 15.0 | Umbral de tiempo para accesibilidad (minutos) |
| `--speed-kmh` | float | 4.5 | Velocidad de caminata (km/h) |
| `--max-homes` | int | 2000 | Número máximo de hogares a considerar |
| `--iterations` | int | 3 | Número de iteraciones de optimización |
| `--generations` | int | 80 | Generaciones por ejecución de NSGA-II |
| `--population` | int | 100 | Tamaño de población de NSGA-II |
| `--categories` | list | health education greens | Categorías de servicios a optimizar |
| `--plot` | flag | False | Generar mapa interactivo HTML |
| `--output-dir` | str | outputs_reordenamiento | Directorio para guardar resultados |

---

## 📁 Archivos de Salida

Después de ejecutar el script, se generarán los siguientes archivos en el directorio `outputs_reordenamiento/`:

### Archivos GeoJSON (Geoespaciales)

1. **homes_initial.geojson**: Ubicaciones iniciales de hogares
2. **homes_optimized.geojson**: Ubicaciones optimizadas de hogares (después del reordenamiento)
3. **services_[categoria]_initial.geojson**: Servicios iniciales por categoría
4. **services_[categoria]_optimized.geojson**: Servicios optimizados por categoría

### Archivos CSV (Datos Tabulares)

5. **optimization_history.csv**: Historial completo de métricas por iteración
6. **comparison_metrics.csv**: Comparación entre estado inicial y final

### Archivos de Visualización

7. **comparison_map.html**: Mapa interactivo comparando estado inicial vs optimizado

---

## 📊 Interpretación de Resultados

### Métricas de Cobertura

El sistema calcula las siguientes métricas:

- **cov_health**: % de hogares con acceso a servicios de salud
- **cov_education**: % de hogares con acceso a servicios educativos
- **cov_greens**: % de hogares con acceso a áreas verdes
- **cov_work**: % de hogares con acceso a zonas de trabajo
- **cov_all**: % de hogares con acceso a TODAS las categorías simultáneamente

### Ejemplo de Salida

```
[ESTADO INICIAL - Métricas de Cobertura]
  cov_health: 0.456 (45.6%)
  cov_education: 0.623 (62.3%)
  cov_greens: 0.389 (38.9%)
  cov_work: 0.512 (51.2%)
  cov_all: 0.234 (23.4%)

[ESTADO FINAL - Métricas de Cobertura]
  cov_health: 0.789 (78.9%)
  cov_education: 0.845 (84.5%)
  cov_greens: 0.678 (67.8%)
  cov_work: 0.723 (72.3%)
  cov_all: 0.567 (56.7%)

Mejora en cov_all: +142.3%
```

### Visualización en el Mapa

El mapa HTML generado muestra:

- **🟢 Verde/Lima**: Hogares con acceso completo (estado inicial/optimizado)
- **🔴 Rojo/Naranja**: Hogares sin acceso completo
- **Círculos de colores**: Servicios por categoría
  - Rojo: Salud
  - Azul: Educación
  - Verde: Áreas verdes
  - Morado: Trabajo

Puedes activar/desactivar capas usando el control en la esquina superior derecha del mapa.

---

## 🔧 Casos de Uso

### Caso 1: Análisis Básico

Analizar la accesibilidad actual de un distrito:

```bash
python ciudad_15min_reordenamiento.py \
    --place "Miraflores, Lima, Peru" \
    --minutes 15 \
    --iterations 1 \
    --plot
```

### Caso 2: Optimización Intensiva

Optimización profunda con más iteraciones:

```bash
python ciudad_15min_reordenamiento.py \
    --place "San Juan de Miraflores, Lima, Peru" \
    --minutes 15 \
    --iterations 5 \
    --generations 120 \
    --population 150 \
    --plot
```

### Caso 3: Enfoque en Salud y Educación

Optimizar solo servicios de salud y educación:

```bash
python ciudad_15min_reordenamiento.py \
    --place "Villa El Salvador, Lima, Peru" \
    --minutes 15 \
    --categories health education \
    --iterations 4 \
    --plot
```

### Caso 4: Ciudad de 10 Minutos

Modelo más restrictivo con umbral de 10 minutos:

```bash
python ciudad_15min_reordenamiento.py \
    --place "Lince, Lima, Peru" \
    --minutes 10 \
    --speed-kmh 5.0 \
    --iterations 3 \
    --plot
```

---

## 🧪 Validación y Verificación

### Verificar Número de Hogares

El sistema garantiza que el número de hogares se mantenga constante. En la salida, verás:

```
[Resultado] Mejor cobertura: 0.678
  Hogares: 2000 (objetivo: 2000)
  Servicios (health): 87
```

Si los hogares no coinciden exactamente, el algoritmo incluye una restricción que penaliza estas soluciones.

### Verificar Mejoras

El archivo `comparison_metrics.csv` muestra la mejora para cada métrica:

```csv
metric,initial,final,improvement,improvement_pct
cov_health,0.456,0.789,0.333,73.0
cov_education,0.623,0.845,0.222,35.6
cov_all,0.234,0.567,0.333,142.3
```

---

## ⚠️ Solución de Problemas

### Problema 1: Error de Memoria

**Síntoma**: `MemoryError` o el proceso se detiene

**Solución**:
```bash
# Reducir número de hogares
python ciudad_15min_reordenamiento.py --place "..." --max-homes 1000

# Reducir población y generaciones
python ciudad_15min_reordenamiento.py --place "..." --population 50 --generations 40
```

### Problema 2: No se Encuentra el Lugar

**Síntoma**: `ValueError: No se pudo geocodificar el lugar`

**Solución**:
- Verificar ortografía del lugar
- Usar formato completo: "Distrito, Ciudad, País"
- Probar con nombre en inglés: "Miraflores, Lima, Peru"

### Problema 3: Datos Insuficientes en OSM

**Síntoma**: Muy pocos servicios u hogares encontrados

**Solución**:
- El sistema tiene un fallback que genera puntos sintéticos
- Considerar contribuir datos a OpenStreetMap para tu área
- Usar un distrito más grande o con mejor cobertura de datos

### Problema 4: Optimización Muy Lenta

**Síntoma**: El proceso tarda demasiado

**Solución**:
```bash
# Configuración rápida
python ciudad_15min_reordenamiento.py \
    --place "..." \
    --max-homes 1000 \
    --iterations 2 \
    --generations 50 \
    --population 60
```

### Problema 5: pymoo No Funciona

**Síntoma**: Error al importar pymoo

**Solución**:
```bash
# Desinstalar y reinstalar versión específica
pip uninstall pymoo
pip install pymoo==0.6.1.1
```

---

## 📚 Fundamentos Técnicos

### Algoritmo NSGA-II

El sistema usa el algoritmo genético NSGA-II (Non-dominated Sorting Genetic Algorithm II) que:

1. Mantiene una población de soluciones
2. Evalúa cada solución según múltiples objetivos
3. Selecciona las mejores usando dominancia de Pareto
4. Genera nuevas soluciones mediante cruce y mutación
5. Converge hacia el frente de Pareto óptimo

### Representación de Soluciones

Cada solución es un vector binario de longitud N (total de ubicaciones):

```
[0, 1, 0, 0, 1, 0, 1, 0, ...]
 ^  ^  ^  ^  ^  ^  ^  ^
 |  |  |  |  |  |  |  |
 Hogar Servicio Hogar Hogar Servicio ...
```

- `0` = La ubicación es un hogar
- `1` = La ubicación es un servicio

### Funciones Objetivo

**f1 = 1 - cobertura**
- Minimizar f1 equivale a maximizar cobertura
- Cobertura = % de hogares con acceso en ≤15 min

**f2 = α * |proporción_servicios - ideal| + β * (1 - otras_coberturas)**
- Penaliza desequilibrios en la proporción de servicios
- Considera el impacto en otras categorías
- α y β son pesos configurables

---

## 🔬 Para Investigadores

### Exportar Datos para Análisis

```python
import geopandas as gpd
import pandas as pd

# Cargar resultados
homes_initial = gpd.read_file('outputs_reordenamiento/homes_initial.geojson')
homes_optimized = gpd.read_file('outputs_reordenamiento/homes_optimized.geojson')
history = pd.read_csv('outputs_reordenamiento/optimization_history.csv')

# Análisis estadístico
print(f"Cobertura inicial: {homes_initial['covered_all'].mean():.3f}")
print(f"Cobertura final: {homes_optimized['covered_all'].mean():.3f}")

# Visualizar evolución
import matplotlib.pyplot as plt
plt.plot(history['cov_all'])
plt.xlabel('Iteración')
plt.ylabel('Cobertura Total')
plt.title('Evolución de la Cobertura')
plt.show()
```

### Modificar Parámetros del Algoritmo

Edita el archivo `ciudad_15min_reordenamiento.py`:

```python
# Línea ~320: Cambiar proporción ideal de servicios
ideal_service_ratio = 0.075  # 7.5% (default)
ideal_service_ratio = 0.10   # 10% (más servicios)

# Línea ~463: Cambiar parámetros de optimización
alpha_balance=0.15  # Peso del balance (default)
alpha_balance=0.20  # Mayor énfasis en balance
```

### Publicaciones y Citas

Si usas este sistema en investigación, considera citar:

```bibtex
@software{sistema_planificacion_urbana_2025,
  title={Sistema de Planificación Urbana con Reordenamiento Dinámico},
  author={Tu Nombre},
  year={2025},
  url={https://github.com/tu-repo}
}
```

---

## 📞 Soporte y Contacto

### Preguntas Frecuentes

**P: ¿Puedo usar datos propios en lugar de OpenStreetMap?**
R: Sí, puedes modificar las funciones `load_services()` y `load_residences()` para cargar tus propios archivos GeoJSON o Shapefiles.

**P: ¿El sistema funciona para ciudades fuera de Perú?**
R: Sí, funciona para cualquier lugar que tenga datos en OpenStreetMap.

**P: ¿Cuánto tiempo tarda el proceso?**
R: Depende del tamaño del área y parámetros. Típicamente:
- Área pequeña (1-2 km²): 10-20 minutos
- Área mediana (3-5 km²): 30-60 minutos
- Área grande (>5 km²): 1-3 horas

**P: ¿Los resultados son implementables en la realidad?**
R: Los resultados son propuestas optimizadas que deben ser evaluadas por urbanistas y considerar restricciones legales, económicas y sociales. El sistema proporciona una base técnica para la toma de decisiones.

### Reporte de Errores

Si encuentras un error:

1. Verifica que tienes la última versión del código
2. Asegúrate de que todas las dependencias estén instaladas
3. Revisa la sección de "Solución de Problemas"
4. Guarda el mensaje de error completo
5. Incluye el comando exacto que usaste

---

## 🎓 Créditos y Licencia

### Basado en:

- **NSGA-II**: Deb, K., et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II"
- **OSMnx**: Boeing, G. (2017). "OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks"
- **Ciudad de 15 minutos**: Moreno, C., et al. (2021). "Introducing the '15-Minute City'"

### Herramientas Utilizadas:

- Python 3.8+
- OSMnx para datos geoespaciales
- NetworkX para análisis de redes
- GeoPandas para procesamiento geoespacial
- pymoo para optimización multi-objetivo
- Folium para visualización de mapas

### Licencia

Este proyecto está bajo licencia MIT. Puedes usarlo, modificarlo y distribuirlo libremente para fines académicos y comerciales.

---

## 🚀 Próximos Pasos

### Para Empezar

1. Instala las dependencias
2. Ejecuta el ejemplo básico con tu distrito
3. Revisa los archivos generados
4. Abre el mapa HTML en tu navegador
5. Analiza las métricas de mejora

### Para Ir Más Allá

1. Experimenta con diferentes números de iteraciones
2. Ajusta los parámetros del algoritmo
3. Prueba con diferentes umbrales de tiempo
4. Compara resultados entre diferentes distritos
5. Integra los resultados en tu tesis o investigación

---

## ✅ Checklist de Verificación

Antes de ejecutar el sistema, verifica:

- [ ] Python 3.8+ instalado
- [ ] Todas las dependencias instaladas (`pip list`)
- [ ] Conexión a Internet activa
- [ ] Espacio en disco suficiente (~500 MB)
- [ ] RAM disponible (mínimo 4 GB)
- [ ] Nombre del lugar verificado en OpenStreetMap

Durante la ejecución, deberías ver:

- [ ] Mensaje de carga del lugar
- [ ] Descarga de red peatonal
- [ ] Carga de servicios (con conteos)
- [ ] Carga de hogares
- [ ] Progreso de optimización (barras de progreso)
- [ ] Métricas de cada iteración
- [ ] Resumen final con mejoras

Al finalizar, verifica que existan:

- [ ] Directorio `outputs_reordenamiento/`
- [ ] Archivos GeoJSON (homes, services)
- [ ] Archivos CSV (history, comparison)
- [ ] Mapa HTML (si usaste --plot)

---

**¡Éxito con tu investigación sobre planificación urbana! 🏙️✨**
