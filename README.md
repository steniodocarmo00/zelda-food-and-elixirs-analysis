# Zelda: Breath of the Wild - Food Analysis AI

## Alunos: Stênio Gabriel Botelho do Carmo, Maria Eduarda Ribeiro Ramos

Este projeto aplica técnicas de Ciência de Dados e Machine Learning para analisar as mecânicas de culinária do jogo *The Legend of Zelda: Breath of the Wild*.

O objetivo é entender matematicamente como os ingredientes influenciam a cura (`Hearts`) e os efeitos especiais (`Effects`), utilizando desde estatística clássica até AutoML.

## Objetivos
* **Regressão:** Prever a quantidade de cura (`Hearts`) baseada nos atributos numéricos e categóricos dos ingredientes.
* **Classificação:** Prever se um ingrediente gera efeitos especiais (ex: *Hasty*, *Mighty*) com base em sua categoria.

## Tecnologias e Dependências
Este projeto foi desenvolvido e testado especificamente com:

* **Python 3.11.6**
* **Jupyter Notebook**

**Bibliotecas Principais:**
* `pandas` & `numpy` (Manipulação de dados)
* `matplotlib` & `seaborn` (Visualização)
* `scipy` & `statsmodels` (Testes estatísticos e inferência)
* `scikit-learn` (Machine Learning e Pipelines)
* `pycaret` (AutoML e Validação de Modelos)

## Resultados Chave

### 1. Classificação (Sucesso Absoluto)
* **Acurácia:** 100%
* **Descoberta:** A mecânica de efeitos especiais é **determinística**. Conseguimos provar que a presença de efeito é definida exclusivamente pela categoria do ingrediente (Regra de Negócio), sem aleatoriedade.

### 2. Regressão (Insight de Limitação)
* **RMSE:** ~1.70 (Melhor modelo: Huber Regressor via PyCaret)
* **Descoberta:** A quantidade de cura não segue uma progressão linear suave baseada na duração ou tipo. Modelos complexos não superaram significativamente o *Baseline* (média simples), indicando que a cura funciona por "tiers" discretos, e não por uma fórmula contínua.

## Como Executar

1. **Clonar o repositório:**
   ```bash
   git clone [https://github.com/SEU_USUARIO/NOME_DO_REPO.git](https://github.com/SEU_USUARIO/NOME_DO_REPO.git)
   cd NOME_DO_REPO
   ```
2. **Configurar o ambiente (Recomendado):** Para garantir a compatibilidade com o PyCaret, recomenda-se usar o Python 3.11.6.
    ```bash
   python -m venv .venv
   source .venv/bin/activate  # No Windows: .venv\Scripts\activate
   ```
3. **Instalar dependências:**
    ```bash
   pip install -r requirements.txt
   ```
4. **Executar a análise:**
    ```bash
   jupyter notebook
   ```

## Estrutura do Projeto
* **data/**: Contém o dataset zelda_food.csv
* **notebooks/**: Contém o notebook com toda a análise (EDA, Modelagem, Otimização).
* **requirements.txt**: Lista de bibliotecas para reprodução

* ## Fonte dos Dados e Licença

* **Dataset:** Zelda Complete Food and Elixirs Dataset
* **Fonte:** [Kaggle - Pavlos Nigur](https://www.kaggle.com/datasets/pavlosnigur/zelda-botw-food-complete)
* **Licença:** CC0: Public Domain (Domínio Público)
* **Nota Legal:** Este projeto utiliza dados de *The Legend of Zelda: Breath of the Wild* para fins estritamente educacionais e de pesquisa. Todos os direitos sobre a propriedade intelectual do jogo pertencem à Nintendo.
