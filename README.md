# MGeo

É um pacote de *Simulação Numérica* para métodos geofísicos eletromagnéticos, onde simula fontes de *Magnetotelúrico* e *marine Controlled Source EM*. O pacote conta com um gerador de Malha em Octree.

## Instalação:

A instalação obedece uma sequência bem definida como mostra os itens abaixo. O usuário deve seguir o passo a passo como mostrado aqui no Readme. 

1. Baixar pacote MGeo no **botão verde** no topo desta página:  
> *https://github.com/projetoemBR/teste_mgeo*

2. Criar pasta MGeo e colocar os arquivos do pacote baixado do item 1.  

3. Dentro da pasta MGeo, criar um ambiente conda utilizando o arquivo .yml:  
> *conda env create -f environment.yml*  
   - **Observação 01:** Essa etapa demora um pouco, pois o CONDA baixa e instala os arquivos especificados dentro do arquivo *environment.yml*.  
   - **Observação 02:** O arquivo *environment.yml* contém todas as especificações e versões dos softwares para que o pacote funcione corretamente.
   - **Observação 03:** Ainda dentro do arquivo *environment.yml*; a primeira linha mostra o nome que será dado ao ambiente. O usuário pode mudar isso sem nenhum problema. As demais linhas são importantes para o funcionamento do pacote.
 
4. Após ambiente criado, ativar o ambiente **mgeo**:  
> *conda activate mgeo*

5. Com o ambiente ativado, instalar pacote Mgeo:  
> *python setup.py install*

Após seguir os passos acima, o pacote MGeo estará instalado dentro de um ambiente Conda.

## Geração de modelos

O Front-End é responsável por gerar Modelo e Malha. Nessa parte pacote irá gerar um arquivo chamado *model.out* que contém todas as informações necessárias para o módulo de *Simulação Numérica* efetuar os cálculos.
