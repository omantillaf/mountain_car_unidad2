# MountainCar-v0 — Aprendizaje por Refuerzo: Q-Learning Tabular vs. Deep Q-Network (DQN)

Resolución del entorno **`MountainCar-v0`** de Gymnasium mediante dos enfoques de aprendizaje por
refuerzo: **Q-Learning tabular** (con discretización del espacio de estados) y **Deep Q-Network
(DQN)** (con aproximación de funciones mediante una red neuronal). El proyecto entrena ambos
agentes, grafica sus curvas de aprendizaje y compara su desempeño.

> **El reto de MountainCar:** un coche sin potencia suficiente para subir la colina de frente debe
> aprender a **balancearse** —retroceder para ganar impulso en la ladera opuesta— y alcanzar la
> bandera. Cada paso otorga una recompensa de **−1**, y el episodio se **trunca a los 200 pasos**.
> Por eso las recompensas son negativas: **más cercanas a 0 = mejor** (llegó a la meta en menos
> pasos); **−200 = fracaso** (nunca llegó).

![El coche alcanzando la bandera en el entorno MountainCar-v0](assets/pygame_meta.jpeg)

---

## Índice

1. [Estructura del repositorio](#estructura-del-repositorio)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Cómo ejecutarlo](#cómo-ejecutarlo)
4. [El proceso, paso a paso](#el-proceso-paso-a-paso)
5. [Esquema del entrenamiento — Q-Learning](#esquema-del-entrenamiento--q-learning-tabular)
6. [Esquema del entrenamiento — DQN](#esquema-del-entrenamiento--dqn)
7. [Evidencia del mejor resultado — Q-Learning](#evidencia-del-mejor-resultado--q-learning-tabular)
8. [Evidencia del mejor resultado — DQN](#evidencia-del-mejor-resultado--dqn)
9. [Cuadro comparativo](#cuadro-comparativo)
10. [Referencias](#referencias)

---

## Estructura del repositorio

```
.
├── README.md                       # Este documento
├── mountain_car_unidad2.ipynb      # Notebook con todo el código y la ejecución
└── assets/
    ├── diagrama_qlearning.svg      # Esquema propio del ciclo Q-Learning
    ├── diagrama_dqn.svg            # Esquema propio del ciclo DQN
    ├── curva_qlearning.png         # Curva de recompensa (evidencia Q-Learning)
    ├── curva_dqn.png               # Curva de recompensa (evidencia DQN)
    └── pygame_meta.jpeg            # Captura del agente alcanzando la meta
```

---

## Requisitos e instalación

- **Python 3.11** (el notebook se ejecutó en 3.11).
- Las dependencias se instalan desde las primeras celdas del notebook, pero también pueden
  instalarse manualmente:

```bash
pip install gymnasium "gymnasium[classic-control]"
pip install numpy matplotlib
pip install pygame            # solo para la visualización con ventana
pip install tensorflow        # solo para el agente DQN
```

| Librería | Uso |
| :--- | :--- |
| `gymnasium` | Entorno `MountainCar-v0` |
| `numpy` | Tabla Q, discretización, vectores de estado |
| `matplotlib` | Curvas de aprendizaje |
| `pygame` | Render de la ventana del agente entrenado (`render_mode="human"`) |
| `tensorflow` / `keras` | Red neuronal del agente DQN |

> **Nota (macOS/Linux con zsh):** el corchete de `gymnasium[classic-control]` debe ir entre
> comillas (`"gymnasium[classic-control]"`), de lo contrario zsh lo interpreta como patrón de
> archivos y falla con `no matches found`.

---

## Cómo ejecutarlo

1. Abrir `mountain_car_unidad2.ipynb` en Jupyter, VS Code o Google Colab.
2. Ejecutar las celdas **en orden, de arriba hacia abajo**. El flujo es:
   - **Paso 2 — Q-Learning tabular:** discretización → entrenamiento (4000 episodios) →
     visualización opcional con `pygame` → curva de recompensa.
   - **Paso 3 — DQN:** construcción de la red principal y la red objetivo → *replay buffer* →
     entrenamiento (300 episodios).
   - **Paso 4 — Resultados:** curva DQN + cuadro comparativo.
3. **Tiempos aproximados:** el Q-Learning tabular corre en **segundos**; el DQN toma **varios
   minutos** en CPU, porque calcula gradientes en cada paso del entorno.

> La celda de visualización con `pygame` abre una ventana externa. Es **opcional**; si se ejecuta
> en un entorno sin pantalla (servidor/Colab), omítela o usa `render_mode="rgb_array"`.

---

## El proceso, paso a paso

### Q-Learning tabular

1. **Discretización.** El espacio de observación es continuo (posición, velocidad). Los métodos
   tabulares requieren estados finitos, así que se divide en una cuadrícula de **20×20 bins** con
   `np.linspace`, y `np.digitize` mapea cada estado continuo a un par de índices enteros.
2. **Tabla Q.** Se inicializa en ceros con forma `(20, 20, 3)` — 3 acciones: izquierda, nada,
   derecha.
3. **Política ε-greedy.** Con probabilidad ε el agente **explora** (acción aleatoria); si no,
   **explota** (`argmax` de la tabla Q). ε arranca en 1.0 y decae ×0.995 por episodio hasta 0.01.
4. **Actualización de Bellman.** En cada paso se corrige el valor:
   `Q(s,a) ← Q(s,a) + α·[r + γ·max Q(s',a') − Q(s,a)]`, con α = 0.1 y γ = 0.99.
5. **Registro y gráfica.** Se guarda la recompensa por episodio y se suaviza con una media móvil.

### Deep Q-Network (DQN)

1. **Aproximación de funciones.** En lugar de discretizar, una red neuronal densa
   (**2 → 24 → 24 → 3**, ReLU en las ocultas y salida lineal) recibe el estado continuo y predice
   el valor Q de cada acción. Optimizador **Adam** (lr = 0.001), pérdida **MSE**.
2. **Red dual (principal + objetivo).** La *red principal* decide y se entrena en cada paso; la
   *red objetivo* es una copia estable que provee el valor futuro, y se **sincroniza cada 10
   episodios**. Esto evita el "blanco móvil" que desestabiliza el entrenamiento.
3. **Experience Replay.** Las transiciones `(s, a, r, s', terminado)` se guardan en un buffer
   (`deque`, máx. 20 000). El entrenamiento toma **minibatches aleatorios de 64** para romper la
   correlación entre estados secuenciales.
4. **Optimización de Bellman.** Para cada muestra del minibatch, el objetivo es
   `y = r + γ·max Q_objetivo(s',a')` (o solo `r` si el estado es terminal), y la red principal
   ajusta sus pesos por retropropagación hacia ese objetivo.

---

## Esquema del entrenamiento — Q-Learning tabular

Ciclo **estado → acción → recompensa → actualización**:

![Esquema del ciclo de entrenamiento de Q-Learning](assets/diagrama_qlearning.svg)

El agente consulta la Tabla Q para decidir (1) la acción; el entorno devuelve (2) el nuevo estado y
(3) la recompensa; con esos datos el agente (4) actualiza la Tabla Q mediante la ecuación de
Bellman. El ciclo se repite en cada paso de cada episodio.

## Esquema del entrenamiento — DQN

Ciclo con **replay buffer, red objetivo y actualización de Bellman**:

![Esquema del ciclo de entrenamiento de DQN](assets/diagrama_dqn.svg)

La red principal (1) ejecuta una acción; la transición se (2) guarda en el *replay buffer*; de ahí
se (3) muestrea un minibatch; la red objetivo (4) estima el valor futuro que alimenta la
optimización de Bellman; esta (5) ajusta los pesos de la red principal por retropropagación; y
periódicamente (6) los pesos se copian a la red objetivo.

---

## Evidencia del mejor resultado — Q-Learning tabular

![Curva de recompensa del agente Q-Learning tabular](assets/curva_qlearning.png)

**Progreso registrado durante el entrenamiento (promedio de los últimos 500 episodios):**

| Episodio | Recompensa promedio | ε |
| ---: | ---: | ---: |
| 500 | −200.0 | 0.082 |
| 1000 | −198.8 | 0.010 |
| 2000 | −184.9 | 0.010 |
| 3000 | −162.4 | 0.010 |
| 4000 | **−153.5** | 0.010 |

**Mejor resultado:** el promedio móvil (ventana 100) alcanza su punto más alto en **≈ −137** cerca
del episodio 3650, con episodios individuales que superan **−120**. El promedio final se estabiliza
en **−153.5**.

**Comentario.** La curva muestra el patrón clásico del refuerzo tabular: durante la fase de
exploración (ε alto) el agente se estanca en −200 —se trunca sin llegar a la meta—, pero conforme ε
decae y explota la Tabla Q, el promedio móvil sube de forma clara y sostenida. El agente **sí
resolvió la tarea**: aprendió la estrategia física de retroceder para ganar impulso y alcanzar la
bandera. La discretización 20×20 resultó suficiente para capturar la dinámica del entorno.

## Evidencia del mejor resultado — DQN

![Curva de recompensa del agente DQN](assets/curva_dqn.png)

**Mejor resultado medido:** el **mejor episodio individual fue −166** (episodio 272). De los 300
episodios, solo **6 superaron −200**, todos en la parte final del entrenamiento (episodios 271–299);
el promedio móvil (ventana 15) apenas llegó a **≈ −195**.

**Comentario (honesto sobre la evidencia).** La curva evidencia que el DQN **apenas comenzó a
aprender** al final de la corrida: se mantuvo en −200 durante ~270 episodios y solo en los últimos
~30 empezó a producir episodios exitosos. Esto es coherente con la teoría —el DQN es más costoso y
tarda más en despegar— pero **300 episodios no bastaron** para que convergiera al nivel del agente
tabular. Para acercarlo a recompensas de −110/−150 habría que **entrenar más episodios** y,
típicamente, ayudar con *reward shaping* (p. ej. premiar la velocidad/altura ganada), dado que la
recompensa −1 constante de MountainCar es notoriamente escasa para el aprendizaje profundo.

> **Nota:** la curva y los números anteriores provienen directamente de la salida ejecutada en el
> notebook. Sirven como registro fiel de esta corrida concreta; una nueva ejecución con más
> episodios debería mejorar el resultado del DQN.

---

## Cuadro comparativo

| Criterio | Q-Learning Tabular | Deep Q-Network (DQN) |
| :--- | :--- | :--- |
| **Representación** | Tabla Q sobre estados discretizados (20×20) | Red neuronal sobre el estado continuo |
| **Estabilidad** | Convergencia predecible una vez ajustados los *bins* | Más sensible a fluctuaciones; depende del *replay* y la red objetivo |
| **Velocidad** | Miles de iteraciones, pero pocos segundos de reloj | Menos iteraciones esperadas, pero mucho tiempo de cómputo por paso |
| **Resultado en esta corrida** | ✅ Resolvió la tarea (mejor prom. ≈ −137; final −153.5) | ⚠️ Empezó a aprender al final (mejor episodio −166; requiere más episodios) |
| **Generalización** | Limitada por la resolución de la discretización | Generaliza en todo el espacio continuo sin discretizar |
| **Dificultad principal** | Diseñar bien la discretización | Ajustar hiperparámetros (lr, tamaño de memoria, sincronización) |

**Conclusión.** Para un problema de baja dimensión como MountainCar-v0, el **Q-Learning tabular fue
más eficiente y efectivo** dentro del presupuesto de cómputo usado: convergió en segundos y resolvió
la tarea. El **DQN** es más general y elimina la discretización manual, pero paga ese poder con un
costo computacional mucho mayor y una necesidad de más episodios para converger, algo que quedó
patente al comparar las 4000 iteraciones baratas del tabular con las 300 iteraciones caras del DQN.

---

## Referencias

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2.ª ed.,
  Caps. 4–6). MIT Press.
- Farama Foundation. *Gymnasium Documentation — MountainCar-v0.*
- Lapan, M. (2020). *Deep Reinforcement Learning Hands-On* (2.ª ed.). Packt.
