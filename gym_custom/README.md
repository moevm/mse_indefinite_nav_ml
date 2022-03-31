# Ray trainer
Для запуска обучения:
```python train.py```

## train.py
Конфиг для создания duckietownEnv/Simulator, для которого необходимо название карты, размер изображения, максимальное количество шагов одной итерации.

```
config = {
        "seed": random.randint(0, 100000),
        "map_name": "loop_empty",
        "max_steps": 5000,
        "camera_width": 640,
        "camera_height": 480,
        "accept_start_angle_deg": 40,
        "full_transparency": True,
        "distortion": True,
        "domain_rand": False
    }
```
Ray конфиг, в котором содержится основная информация системы, какое количество ядер/видеокарт/памяти необходимо запросить.
```
ray_init_config = {
        "num_cpus": 8,
        "num_gpus": 0,
        "object_store_memory": 78643200,  # 8 * 1024 * 1024
    }
```
Конфигурация кластера: Название среды, количество видеокарт, количество воркеров (-1 от количества ядер), количество gpu для одного воркера, количество сред для одного воркера, какой фреймворк использовать для обучения.
```
rllib_config.update({
        "env": ENV_NAME,
        "num_gpus": 1,
        "num_workers": 8,  # 32,
        "gpu_per_worker": 1,
        "env_per_worker": 2,
        "framework": "torch",
        "lr": 0.0001,
    })
```
Конфигурация trainer:
```PPOTrainer``` - алгоритм обучения PPO, 
```stop``` - параметр, после которого необходимо закончить обучение,
```checkpoint_at_end``` - сохранять ли модель в конце обучения,
```config``` - config,
```keep_checkpoints_num``` - количество моделей для сохранения,
```checkpoint_score_attr``` - параметр, после которого надо сохранять новую модель(если награда за эпизод выше предыдущих)

```
tune.run(
        PPOTrainer,
        stop={"timesteps_total": 2000000},
        checkpoint_at_end=True,
        config=conf,
        trial_name_creator=lambda trial: trial.trainable_name,
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        checkpoint_score_attr="episode_reward_mean"
    )
```

## env.py

```
class Environment
```
Класс, ответственный за создание окружения, конструктор принимает конфиг, в зависимости от значений которого будет создаваться env.
Метод ```create_env``` создает окружение по-заданному конфигу. Метод ```wrap``` добавляет врапперы для окружения (меняет стандартный функционал окружения).

