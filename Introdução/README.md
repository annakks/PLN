* EXERCÍCIOS DE PROCESSAMENTO DE LINGUAGEM NATURAL *

1. INTRODUÇÃO 

TERMINOLOGIA E CONCEITOS

TC.1.1. Selecione uma obra literária de domínio público (ex. livros tais como Vinte mil léguas submarinas (de Júlio Verne), a Bíblia, etc.) e ilustre a variedade de dados presente. Considere, por exemplo, a construção de frases, orações etc. e compare com expressões de uso corrente.

Na obra Os Lusíadas (Luís de Camões) escrita no século 16 no gênero poesia épica, sendo que todos os versos heróicos decassílabos de 8º rima (ABABABCC), influência das obras gregas Ilíadas (Aquiles) e Odisseia (Ulisses). Pela sua estrutura, ritmo e forma, levou cerca de 12 anos para ser concluída, além da importância histórica que envolve mitologia grega.
Já no primeiro verso “As armas e os barões assinalados” em que apresenta, em portugues clássico, os personagens da epopéia, em adaptação com expressões corrente “Os guerreiros ilustres e renomados”, na construção da frase, sintaxe no verso "Por mares nunca de antes navegados/Passaram ainda além da Taprobana" a ordem das palavras não segue uma estrutura de fácil compreensão diferente se reescrevermos “Navegaram por mares que nunca tinham sido navegados antes e alcançaram um lugar além da Taprobana”



TC.1.2. Exemplifique uma sentença, escrita na língua portuguesa, que pode surgir em um site de pré-atendimento em uma concessionária, que potencialmente seja difícil de ser interpretado por um chatbot. Explique sua resposta em termos de estruturação da sentença e suponha que ela esteja gramaticalmente correta.

Ao usar a frase  “Mais impostos, Faz o L!’”, uma pessoa consegue entender toda o sarcasmo/ironia que tem na frase, porém o chatbot apesar de tentar deduzir têm dificuldade de reconhecer o slogan informal do atual presidente LULA, que se transformou em um meme usado de forma irônica, em crítica a medidas do governo atual. 
Isso acontece porque os chatbots não atualizados com a rapidez com que as informações acontecem, existe também frases ambíguas ou mal formuladas, uso de gírias e jargões regionais.   


TC. 1.3. Sistemas de PLN são geralmente compostos por modelos que são treinados utilizando corpora de texto. Por que modelos que são válidos hoje podem não mais ser adequados daqui a dois anos?

Os usuários/sociedades mudam conforme o tempo e região uma modelo pode se basear em uma corpora que “atende” determinada sociedade e em 2 anos esta pode evoluir culturalmente sofrendo variações linguísticas e uma palavra pode ter seu significado alterado tornando os modelos utilizados ultrapassados.

TC.1.4. Por que a utilização de emojis ou outros símbolos não presentes na linguagem textual formal podem dificultar a operação de um sistema de PNL?

Você usa o emoji🙏como representação de um bater de palmas ou como emoji🙏 representa oração ou agradecimento?? Se é difícil para humanos entender, imagina uma máquina. Emojis e símbolos podem ter diferentes significados em ambientes, cidades, estados ou Países diferentes.


TC.1.5. Dê um exemplo de sentença em um processo comunicativo onde os referentes considerados pelo transmissor e pelo receptor podem ser distintos caso não haja adequada contextualização do processo comunicativo.

O Solicitante acertou o homem com uma faca.

O Solicitante informa que o indivíduo agrediu a sua mãe.

Ambas as frases a pessoa que está passando a mensagem pode estar passando uma determinada informação, o solicitante estar armado com uma faca e acertou o homem e a mãe do sol foi agredida por um indivíduo e o receptor pode estar entendendo que o homem armado com uma faca foi acertado pelo solicitante e o indivíduo agrediu a própria mãe.

TC.1.6. Exemplifique uma saída para o processo de lematização e stemização. Considere a seguinte sentença:
“Assim que amanheceu, os estudantes, apressados, acordaram e saíram correndo para fazer a prova”.

Lematização e stemização são técnicas para reduzir as palavras de tal forma que chegue a sua raiz, ou seja, seu menor elemento no qual deu origem a esta e outras palavras. 
A Lematização e a stemização diferem em seu resultado, na Lematização a redução respeitará a existência da raiz na gramática, que recebe o nome de lemma, já na stemização, a raiz que recebe o nome de stem pode chegar a uma raiz inexistente gramaticalmente, mas ainda com valor para análise. 

Exemplo:

“Assim que amanheceu, os estudantes, apressados, acordaram e saíram correndo para fazer a prova”.

Lematização:

“Assim que amanhecer, estudante, pressa, acordar sair correr para fazer a prova”.


Stemização:

“Assim q amanhec, estud, press, acord e sai corr para faz a prov”.

 
TC.1.7. Cite dois possíveis usos das tags do tipo POS. Forneça exemplos com sentenças simples, expressas na língua portuguesa ou inglesa.

É legal, mas deixou a desejar!

[(É, 'AUX'), (legal, 'ADJ'), (,, 'PUNCT'), (mas, 'CCONJ'), (deixou, 'VERB'), (a, 'SCONJ'), (desejar, 'VERB'), (!, 'PUNCT')]

Conjunção ('CCONJ): Palavras que ligam palavras ou grupos de palavras, coordenando ou subordinando frases, algumas podem ajudar a entender negativas sem palavras padrões como o “não” . Exemplos: mas, porém, todavia, entretanto.

Pontuação (PUNCT): Apesar de não ser uma categoria gramatical convencional, ajuda a estruturar o texto, indicar pausas, enfatizar ideias e clarificar o significado. Exemplos: !!!, ;, ?
