import pandas as pd
import numpy as np

np.random.seed(42)

n = 1500  # количество строк (больше 1000 как требует задание)

data = {
    "age": np.random.randint(18, 60, n),
    "gender": np.random.choice(["Male", "Female"], n),
    "salary": np.random.randint(800, 5000, n),
    "city": np.random.choice(["Almaty", "Astana", "Shymkent"], n),
    "department": np.random.choice(["IT", "HR", "Finance", "Marketing"], n),
    "experience": np.random.randint(0, 20, n),
    "score": np.random.randint(50, 100, n),
}

df = pd.DataFrame(data)

# добавим дату
df["date"] = pd.date_range(start="2023-01-01", periods=n)

# добавим missing values (очень важно для задания)
df.loc[np.random.choice(n, 200), "salary"] = np.nan
df.loc[np.random.choice(n, 100), "city"] = np.nan

# сохраняем
df.to_csv("sample_data/sample1.csv", index=False)
df.to_excel("sample_data/sample2.xlsx", index=False)

print("Datasets created!") 
