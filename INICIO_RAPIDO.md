# 🚀 Guía de Inicio Rápido

## Instalación y Ejecución en 5 Minutos

### Paso 1: Instalar Dependencias (2 minutos)

```bash
pip install -r requirements.txt
```

**Verifica la instalación:**
```bash
python -c "import osmnx, pymoo, geopandas; print('✅ Todo instalado correctamente')"
```

### Paso 2: Ejecutar Ejemplo Rápido (3 minutos)

**Opción A - Usando el script de ejemplo:**
```bash
python ejemplo_rapido.py
```

**Opción B - Comando directo:**
```bash
python ciudad_15min_reordenamiento.py --place "San Juan de Miraflores, Lima, Peru" --minutes 15 --plot
```

### Paso 3: Ver Resultados

Los resultados se guardan en la carpeta `outputs_reordenamiento/` (u `outputs_ejemplo_rapido/`).

**Archivos importantes:**
- 📄 `comparison_map.html` - Abre este archivo en tu navegador para ver el mapa interactivo
- 📊 `comparison_metrics.csv` - Tabla con la comparación de métricas
- 📈 `optimization_history.csv` - Evolución de las métricas por iteración

---

## Comandos Útiles

### Análisis Básico (5-10 minutos)
```bash
python ciudad_15min_reordenamiento.py \
    --place "Miraflores, Lima, Peru" \
    --minutes 15 \
    --max-homes 1500 \
    --iterations 2 \
    --plot
```

### Análisis Completo (30-60 minutos)
```bash
python ciudad_15min_reordenamiento.py \
    --place "San Juan de Miraflores, Lima, Peru" \
    --minutes 15 \
    --max-homes 2500 \
    --iterations 4 \
    --generations 100 \
    --population 120 \
    --categories health education greens work \
    --plot
```

### Solo Salud y Educación (10-15 minutos)
```bash
python ciudad_15min_reordenamiento.py \
    --place "Villa El Salvador, Lima, Peru" \
    --minutes 15 \
    --categories health education \
    --iterations 3 \
    --plot
```

---

## Interpretación Rápida de Resultados

### En la Terminal

Busca esta sección al final de la ejecución:

```
[COMPARATIVA FINAL]
metric              initial   final  improvement  improvement_pct
cov_health          0.456    0.789       0.333          73.0
cov_education       0.623    0.845       0.222          35.6
cov_all             0.234    0.567       0.333         142.3
```

**¿Qué significa?**
- `cov_all` pasó de 23.4% a 56.7% → Mejora de +142.3%
- Ahora 56.7% de los hogares tienen acceso a TODOS los servicios en ≤15 minutos

### En el Mapa HTML

1. **Abre** `comparison_map.html` en tu navegador
2. **Activa/Desactiva capas** usando el control en la esquina superior derecha
3. **Compara**:
   - 🔴 Estado Inicial (círculos pequeños)
   - 🟢 Estado Optimizado (círculos más grandes)

**Interpretación:**
- Más puntos verdes/lima = Mejor cobertura
- Los servicios se han reubicado para cubrir más hogares
- El número total de hogares se mantiene constante

---

## Problemas Comunes y Soluciones

### ❌ Error: "No se pudo geocodificar el lugar"

**Solución:** Verifica el nombre del lugar
```bash
# ✅ Correcto
--place "Miraflores, Lima, Peru"

# ❌ Incorrecto
--place "Miraflores"
--place "Miraflores Lima"
```

### ❌ Error: MemoryError

**Solución:** Reduce el número de hogares
```bash
--max-homes 1000  # En lugar de 2000
```

### ❌ El proceso es muy lento

**Solución:** Usa configuración rápida
```bash
--max-homes 1000
--iterations 2
--generations 40
--population 60
```

### ❌ Error al importar pymoo

**Solución:**
```bash
pip uninstall pymoo
pip install pymoo==0.6.1.1
```

---

## Checklist de Verificación

Antes de ejecutar, verifica:

- [ ] Python 3.8 o superior instalado
- [ ] Todas las dependencias instaladas (`pip list | grep osmnx`)
- [ ] Conexión a Internet activa
- [ ] Al menos 4 GB de RAM disponible
- [ ] ~500 MB de espacio en disco disponible

Durante la ejecución, deberías ver:

- [ ] Mensaje: "Cargando datos geográficos..."
- [ ] Mensaje: "Cargando red peatonal..."
- [ ] Barras de progreso de NSGA-II
- [ ] Métricas después de cada iteración

Al finalizar:

- [ ] Directorio `outputs_reordenamiento/` creado
- [ ] Múltiples archivos .geojson presentes
- [ ] Archivos .csv con métricas
- [ ] Archivo `comparison_map.html` (si usaste --plot)

---

## Próximos Pasos

1. ✅ **Ejecuta el ejemplo rápido** para familiarizarte con el sistema
2. 📊 **Analiza los resultados** en los archivos CSV y el mapa HTML
3. 🔧 **Ajusta parámetros** según tus necesidades
4. 📈 **Compara diferentes configuraciones** para tu investigación
5. 📝 **Documenta tus hallazgos** para tu tesis

---

## Recursos Adicionales

- 📖 **README.md completo** - Documentación detallada
- 🎓 **Fundamentos teóricos** - Ver sección en README.md
- 🔬 **Para investigadores** - Guía de análisis avanzado en README.md
- ⚠️ **Solución de problemas** - Sección completa en README.md

---

## Ayuda Rápida

**¿Necesitas ayuda?**

1. Consulta el README.md completo
2. Verifica la sección de "Solución de Problemas"
3. Revisa que todas las dependencias estén correctamente instaladas
4. Intenta con un lugar diferente (con mejor cobertura de OSM)

**¿Todo funcionó?** ¡Excelente! Ahora puedes:
- Experimentar con diferentes lugares
- Ajustar parámetros del algoritmo
- Integrar los resultados en tu tesis
- Realizar análisis comparativos entre distritos

---

**¡Éxito con tu investigación! 🎯**
