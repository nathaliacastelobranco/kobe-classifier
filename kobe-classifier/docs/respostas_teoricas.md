# ***Projeto Final - Engenharia de machine learning***
## Nathalia Castelo Branco 

### **1. A solução criada nesse projeto deve ser disponibilizada em repositório git e disponibilizada em servidor de repositórios (Github (recomendado), Bitbucket ou Gitlab). O projeto deve obedecer o Framework TDSP da Microsoft. Todos os artefatos produzidos deverão conter informações referentes a esse projeto (não serão aceitos documentos vazios ou fora de contexto). Escreva o link para seu repositório.**

> [Repositório - Kobe Classifier Infnet](https://github.com/nathaliacastelobranco/kobe-classifier)


### **2. Iremos desenvolver um preditor de arremessos usando duas abordagens (regressão e classificação) para prever se o "Black Mamba" (apelido de Kobe) acertou ou errou a cesta. Para começar o desenvolvimento, desenhe um diagrama que demonstra todas as etapas necessárias em um projeto de inteligência artificial desde a aquisição de dados, passando pela criação dos modelos, indo até a operação do modelo.**

![Diagrama de um projeto de Inteligência Artificial]()

### **3. Descreva a importância de implementar pipelines de desenvolvimento e produção numa solução de aprendizado de máquinas.**

> Os pipelines são encadeamentos de processos, como uma esteira de dados, onde existe um mapeamento das funções e processos relacionados à uma solução de aprendizagem de máquina. A ideia de utilizar pipelines no desenvolvimento de soluções de ML é diretamente relacionado à reprodutibilidade e à ideia de modularização e microserviços. 

> **Exemplificando sua importância:** Um projeto de ML com uma entrada de dados e diversos modelos testados para eleger o modelo vencedor, pode ser dividido em alguns pipelines e caso exista a necessidade de manutenção em um dos serviços, seja o de entrada de dados, seja um modelo existente ou a criação de um novo modelo, encadear todo o processo em pipelines torna o projeto mais fácil de gerir e fazer manutenções.

### **4. Como as ferramentas Streamlit, MLFlow, PyCaret e Scikit-Learn auxiliam na construção dos pipelines descritos anteriormente? A resposta deve abranger os seguintes aspectos:**

- a. Rastreamento de experimentos;
- b. Funções de treinamento;
- c. Monitoramento da saúde do modelo;
- d. Atualização de modelo;
- e. Provisionamento (Deployment).

> Na etapa de rastreamento de experimentos, o MLFlow é a ferramenta mais adequada, visto que oferece suporte à registro e comparação de modelos e métricas.

> Na etapa de funções de treinamento, o Scikit-learn trás muitos algoritmos pré-implementados, fazendo com que o tempo de desenvolvimento do modelo seja reduzido e o foco seja em otimizar os hiperparâmetros. Nesta etapa o PyCaret também tem sua atuação, pois possibilita testes de performance de modelos de maneira otimizada para um pré-projeto.

> Já na etapa de monitoramento de saúde do modelo, o MLFlow também tem seu destaque, visto que podemos acompanhar o desempenho do modelo. Nesta etapa também podemos construir visualizações personalizadas com o Streamlit para identificar possíveis mudanças no comportamento do modelo.

> Para a atualização do modelo, o MLFlow permite identificar os melhores hiperparâmetros, além de realizar o treinamento e identificar o desempenho do modelo atualizado.

> No caso do deploy, o MLFlow pode ser servido através de containers ou utilizado em conjunto com outras ferramentas como o Kedro (conforme esse trabalho). O Streamlit possui uma cloud pública que pode ser utilizada para criação de dashboards de acompanhamentos de modelos via web. 

### **5. Com base no diagrama realizado na questão 2, aponte os artefatos que serão criados ao longo de um projeto. Para cada artefato, indique qual seu objetivo.**

> No TDSP da microsoft, os artefatos são os documentos escritos (em verde no gráfico). Os obrigatórios são o Project Charter, que é a visão do todo (o diagrama da questão 2) e o Project Final Report, que é a dashboard gerada no Streamlit.

> * Project Charter - Criado inicialmente para gerir e direcionar a construção do projeto de machine learning, foi desenvolvido um diagrama conforme a questão 2
> * Project Final Report - Criado para visualizar e monitorar a implantação planejada no Project Charter

> Para os artefatos dos pipelines e no MLflow, os arquivos de dados são considerados como artefatos.

### **6.Implemente o pipeline de processamento de dados com o mlflow, rodada (run) com o nome "PreparacaoDados":**

> Localização do pipeline implementado: kobe-classifier\src\kobe_classifier\pipelines\PreparacaoDados

**a. Dados baixados na pasta:** 
> data/01_raw

**b. Dimensão resultante do dataset:** 
> (20285, 7) e dados salvos em: data/03_primary/data_filtered.parquet

**c. Separe os dados em treino (80%) e teste (20 %) usando uma escolha aleatória e estratificada.**
> Dados separados entre treino e teste pela função split_data(prepared_data, test_size, seed) da pipeline PreparacaoDados.

**d. Registre os parâmetros (% teste) e métricas (tamanho de cada base) no MlFlow**

> Parâmetros de treino e teste definidos em kobe-classifier\conf\base\parameters.yml

### **7. Implementar o pipeline de treinamento do modelo com o Mlflow usando o nome "Treinamento"**

> A 7 foi desenvolvida na pipeline que está em kobe-classifier\src\kobe_classifier\pipelines\Treinamento

### **8. Registre o modelo de classificação e o disponibilize através do MLFlow através de API. Selecione agora os dados da base de dados original onde shot_type for igual à 3PT Field Goal (será uma nova base de dados) e através da biblioteca requests, aplique o modelo treinado. Publique uma tabela com os resultados obtidos e indique o novo log loss e f1_score.**

Disponibilização de API:
> mlflow models serve --env-manager local -m runs:/6a400a36627b495485eb58e6c24ca2ef/knn_model --port 5001

**a. O modelo é aderente a essa nova base? Justifique.**
> Não, conforme visto na matriz de confusão da dashboard do Streamlit, o modelo erra muito mais do que acerta. Além disso, tanto o F1 Score de 0,31 quanto o como o log loss de 15, representam um modelo com uma performance bem ruim.

**b. Descreva como podemos monitorar a saúde do modelo no cenário com e sem a disponibilidade da variável resposta para o modelo em operação**

> Para a situação que possuímos a variável resposta, é possível monitorar a saúde do modelo comparando as predições com os resultados reais. Para isso, é necessário definir a métrica que será avaliada e um valor aceitável para a mesma, como por exemplo a precisão e o F1 Score. Essa avaliação pode ser feita periodicamente, como uma análise de logs.

> Já para a situação em que não temos a variável resposta, o ideal é monitorar os logs e identificar quedas ou picos das métricas utilizadas para avaliação. Caso seja identificado algum problema, uma ação deve ser tomada. 

**c. Descreva as estratégias reativa e preditiva de retreinamento para o modelo em operação.**

> A estratégia reativa é utilizada quando algum problema é detectado no desempenho do modelo e este problema é o gatilho para o retreinamento. É mais econômica computacionalmente, mas depende da ocorrência de erros para iniciar.

> A estratégia preditiva funciona baseada na previsão do desempenho do modelo, sendo ele retreinado periodicamente conforme a predição. É mais cara, pois o modelo pode não ter problemas e ser retreinado devido à previsão de desempenho.

**9. Implemente um dashboard de monitoramento da operação usando Streamlit.**

> Dashboard em: dashboard/dashboard.py