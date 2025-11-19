# 🎯 INSTRUCCIONES DE INSTALACIÓN Y USO

## ¡Bienvenido al Sistema de Planificación Urbana con Reordenamiento Dinámico!

Este sistema te permite optimizar la distribución de hogares y servicios en una ciudad
para maximizar la accesibilidad bajo el concepto de **Ciudad de 15 Minutos**.

---

## 📦 PASO 1: Extrae el archivo ZIP

```bash
# Windows: Click derecho > Extraer todo
# Linux/Mac: 
unzip sistema_planificacion_urbana_v2.zip
cd sistema_planificacion_urbana_v2
```

---

## 🔧 PASO 2: Instala las dependencias

```bash
pip install -r requirements.txt
```

**Verificar instalación:**
```bash
python verificar_instalacion.py
```

Si todo está bien, verás: ✅ TODAS LAS VERIFICACIONES PASARON

---

## 🚀 PASO 3: Ejecuta el ejemplo rápido

**Opción A - Script automático:**
```bash
python ejemplo_rapido.py
```

**Opción B - Comando personalizado:**
```bash
python ciudad_15min_reordenamiento.py \
    --place "San Juan de Miraflores, Lima, Peru" \
    --minutes 15 \
    --plot
```

---

## 📁 PASO 4: Revisa los resultados

Los resultados se guardan en la carpeta `outputs_reordenamiento/` o `outputs_ejemplo_rapido/`

**Archivos importantes:**

1. **comparison_map.html** 
   - Abre este archivo en tu navegador
   - Compara el estado inicial vs optimizado
   - Usa el control de capas para activar/desactivar elementos

2. **comparison_metrics.csv**
   - Tabla con todas las métricas
   - Compara inicial vs final
   - Muestra el porcentaje de mejora

3. **optimization_history.csv**
   - Evolución de las métricas por iteración
   - Útil para gráficos de convergencia

4. **homes_optimized.geojson**
   - Ubicaciones finales de hogares
   - Incluye información de cobertura

5. **services_[categoria]_optimized.geojson**
   - Ubicaciones finales de servicios por categoría

---

## 📖 DOCUMENTACIÓN COMPLETA

Revisa estos archivos para más información:

- **INICIO_RAPIDO.md** - Guía de inicio en 5 minutos
- **README.md** - Documentación completa y detallada
- **CHANGELOG.md** - Historial de cambios y mejoras
- **configuracion_avanzada.py** - Parámetros avanzados

---

## ⚠️ SOLUCIÓN DE PROBLEMAS COMUNES

### Problema 1: Error al importar módulos

**Síntoma:** `ModuleNotFoundError: No module named 'osmnx'`

**Solución:**
```bash
pip install -r requirements.txt
# Si hay errores, instala uno por uno:
pip install osmnx==1.9.3
pip install pymoo==0.6.1.1
```

### Problema 2: Error de memoria

**Síntoma:** `MemoryError` o el proceso se detiene

**Solución:**
```bash
# Reduce el número de hogares
python ciudad_15min_reordenamiento.py \
    --place "..." \
    --max-homes 1000 \
    --iterations 2
```

### Problema 3: No se encuentra el lugar

**Síntoma:** `ValueError: No se pudo geocodificar el lugar`

**Solución:**
- Usa el formato completo: "Distrito, Ciudad, País"
- Ejemplo correcto: "Miraflores, Lima, Peru"
- Verifica la ortografía

### Problema 4: Proceso muy lento

**Síntoma:** El proceso tarda más de 30 minutos

**Solución:**
```bash
# Usa configuración rápida
python ciudad_15min_reordenamiento.py \
    --place "..." \
    --max-homes 1000 \
    --iterations 2 \
    --generations 40 \
    --population 60
```

---

## 🎓 CASOS DE USO PARA TU TESIS

### Análisis Básico de Accesibilidad
```bash
python ciudad_15min_reordenamiento.py \
    --place "Tu Distrito, Lima, Peru" \
    --minutes 15 \
    --iterations 3 \
    --plot
```

### Comparación Entre Distritos
```bash
# Distrito 1
python ciudad_15min_reordenamiento.py \
    --place "Miraflores, Lima, Peru" \
    --output-dir outputs_miraflores \
    --plot

# Distrito 2
python ciudad_15min_reordenamiento.py \
    --place "San Juan de Miraflores, Lima, Peru" \
    --output-dir outputs_sjm \
    --plot
```

### Análisis Profundo para Tesis
```bash
python ciudad_15min_reordenamiento.py \
    --place "Tu Distrito, Lima, Peru" \
    --minutes 15 \
    --max-homes 2500 \
    --iterations 5 \
    --generations 120 \
    --population 150 \
    --categories health education greens work \
    --plot
```

---

## 📊 INTERPRETACIÓN DE RESULTADOS

### Métricas Clave

- **cov_health**: % de hogares con acceso a salud en ≤15 min
- **cov_education**: % de hogares con acceso a educación en ≤15 min
- **cov_greens**: % de hogares con acceso a áreas verdes en ≤15 min
- **cov_work**: % de hogares con acceso a zonas de trabajo en ≤15 min
- **cov_all**: % de hogares con acceso a TODAS las categorías

### Ejemplo de Salida

```
[ESTADO INICIAL]
  cov_all: 0.234 (23.4%)

[ESTADO FINAL]
  cov_all: 0.567 (56.7%)

Mejora: +142.3%
```

Esto significa que:
- Inicialmente, solo el 23.4% de hogares tenía acceso completo
- Después de la optimización, el 56.7% tiene acceso completo
- Una mejora de 142.3%

---

## 🔬 PARA INVESTIGADORES

### Exportar Datos para Análisis Estadístico

```python
import pandas as pd
import geopandas as gpd

# Cargar resultados
history = pd.read_csv('outputs_reordenamiento/optimization_history.csv')
comparison = pd.read_csv('outputs_reordenamiento/comparison_metrics.csv')

# Análisis
print(comparison)

# Gráfico de evolución
import matplotlib.pyplot as plt
plt.plot(history['cov_all'])
plt.xlabel('Iteración')
plt.ylabel('Cobertura Total')
plt.title('Evolución de la Cobertura')
plt.savefig('evolucion.png')
```

### Integrar en LaTeX (Tesis)

```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{evolucion.png}
    \caption{Evolución de la cobertura total durante el proceso de optimización}
    \label{fig:evolucion}
\end{figure}
```

---

## ✅ CHECKLIST ANTES DE USAR

- [ ] Python 3.8 o superior instalado
- [ ] Todas las dependencias instaladas
- [ ] Conexión a Internet activa
- [ ] Al menos 4 GB de RAM disponible
- [ ] Nombre del lugar verificado en OpenStreetMap

---

## 📞 AYUDA Y SOPORTE

Si tienes problemas:

1. ✅ Ejecuta `python verificar_instalacion.py`
2. ✅ Revisa la sección de problemas comunes arriba
3. ✅ Consulta el README.md completo
4. ✅ Revisa el CHANGELOG.md para ver cambios recientes

---

## 🎉 ¡TODO LISTO!

Ya estás preparado para usar el sistema. Comienza con:

```bash
python ejemplo_rapido.py
```

O personaliza tu análisis:

```bash
python ciudad_15min_reordenamiento.py --place "Tu Distrito, Lima, Peru" --minutes 15 --plot
```

---

## 🌟 CARACTERÍSTICAS PRINCIPALES

✨ **Intercambio dinámico de hogares y servicios**
- El sistema puede reubicar tanto hogares como servicios para optimizar la distribución

🏠 **Número de hogares constante**
- Se garantiza que el número total de hogares se mantiene durante la optimización

🔄 **Optimización iterativa**
- El proceso se repite para cada categoría de servicio, mejorando gradualmente

📊 **Visualización comparativa**
- Mapa interactivo que muestra el antes y después de la optimización

📈 **Métricas detalladas**
- Cálculo de cobertura por categoría y cobertura total

🎯 **Base en Ciudad de 15 Minutos**
- Optimiza para que todos tengan acceso a servicios esenciales en ≤15 minutos

---

**¡Éxito con tu investigación! 🚀**

Para más información, lee el **README.md** completo.
