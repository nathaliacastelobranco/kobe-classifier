# Projeto Final - Engenharia de machine learning

## 3. Descreva a importância de implementar pipelines de desenvolvimento e produção numa solução de aprendizado de máquinas.

    Os pipelines são encadeamentos de processos, como uma esteira de dados, onde existe um mapeamento das funções e processos relacionados à uma solução de aprendizagem de máquina. A ideia de utilizar pipelines no desenvolvimento de soluções de ML é diretamente relacionado à reprodutibilidade e à ideia de modularização e microserviços. 

    Exemplificando sua importância:
        Um projeto de ML com uma entrada de dados e diversos modelos testados para eleger o modelo vencedor, pode ser dividido em alguns pipelines e caso exista a necessidade de manutenção em um dos serviços, seja o de entrada de dados, seja um modelo existente ou a criação de um novo modelo, encadear todo o processo em pipelines torna o projeto mais fácil de gerir e fazer manutenções.

## 4. Como as ferramentas Streamlit, MLFlow, PyCaret e Scikit-Learn auxiliam na construção dos pipelines descritos anteriormente? A resposta deve abranger os seguintes aspectos:

- a. Rastreamento de experimentos;
- b. Funções de treinamento;
- c. Monitoramento da saúde do modelo;
- d. Atualização de modelo;
- e. Provisionamento (Deployment).