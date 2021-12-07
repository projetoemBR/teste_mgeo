# MGeo

## Instalação:

1. Baixar pacote mgeo:  
> *https://github.com/projetoemBR/teste_mgeo*

2. Criar pasta MGeo e colocar os arquivos do pacote baixado do item 1.  

3. Dentro da pasta MGeo, criar um ambiente conda utilizando o arquivo .yml:  
> *conda env create -f environment.yml*  
   - **Observação 01:** Essa etapa demora um pouco, pois o CONDA baixa e instala os arquivos.  
   - **Observação 02:** O *arquivo environment.yml* contem todas as especificações softwares e suas verões para que o pacode funcione corretamente.
   - **Observação 02:** Ainda dentro do arquivo *arquivo environment.yml*; a primeira linha mostra o nome que será dado ao ambiente. O usuário pode mudar isso sem nenhum problema. As demais linhas são importante para o funcionamento do pacote.
 
4. Após ambiente criado, ativar o ambiente **mgeo**:  
> *conda activate mgeo*

5. Com o ambiente ativado, instalar pacote Mgeo:  
> *python setup.py install*

Done!

