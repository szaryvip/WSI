# „Q-Uber”
# Elon Piżmo konstruuje autonomiczne samochody do swojego
# najnowszego biznesu. Dysponujemy planszą NxN (domyślnie N=8)
# reprezentującą pole do testów jazdy. Na planszy jako
# przeszkody stoją jego bezpłatni stażyści (reprezentują
# dziury o ujemnej punktacji). Mamy dwa autonomiczne
# samochody: Random-car, który kierunek wybiera rzucając
# kością (błądzi losowo po planszy) oraz Q-uber, który uczy
# się przechodzić ten labirynt (używa naszego algorytmu).
# Samochody zaczynają w tym samym zadanym punkcie planszy
# i wygrywają, jeśli dotrą do punktu końcowego, którym jest
# inny punkt planszy. Istnieje co najmniej jedna ścieżka do
# startu do końca. Elon oszczędzał na module do liczenia
# pierwiastków, dlatego samochody poruszają się przy użyciu
# metryki Manhattan (góra, dół, lewo, prawo). Jeżeli
# samochód natrafi na stażystę to kończy bieg i przegrywa.
# Analogicznie jak wejdzie na punkt końca to wygrywa i
# również nie kontynuuje dalej swojej trasy. Celem agenta
# jest minimalizacja pokonywanej trasy.

# Muszą Państwo napisać generator takiego labiryntu
# – wystarczy użyć algorytmu DFS(w głąb) do sprawdzenia czy
# istnieje ścieżka. Oczywiście na potrzebę testów sugeruję
# ustawić stałe ziarno losowania lub wczytywać mapę z pliku.
# Drugą częścią zadania jest implementacja obu agentów
# (tego losowego i tego używającego Q-Learning).

# W raporcie oczekiwałbym, że opiszą Państwo co stanowi
# poszczególne elementy algorytmu Q-Learning (akcje, polityka,
# stany i nagroda). Algorytm ten ma kilka parametrów, więc
# mam nadzieję, że sprawdzą Państwo wpływ zmiany parametrów
# na działanie algorytmu. Trudno mi sobie wyobrazić jakąś
# sensowną wizualizację dla zadania, ale gdyby byli Państwo
# w stanie wykazać w raporcie, że model nauczył się to
# byłoby wspaniale (może przykładowe wartości parametrów
# i sekwencja stanów dla przypadku nienauczonego i nauczonego
# albo użyć pygame z zadania o grach deterministycznych?).
# Na pewno kluczowe będzie pokazanie, że agent nauczył się
# najkrótszej trasy do celu.

# - generacja planszy powinna być wykonywana do skutku.
# - co do wartości startu i początku najlepiej jest je
# przekazać jako parametry waszego rozwiązania
# - dziury na planszy lepiej jest zdefiniować jako
# prawdopodobieństwo ich wystąpienia (tzn. współczynnik)
# niż wartość bezwzględną.
# - porażki agenta losowego nie powinny wpływać
# na proces nauki drugiego agenta. Innymi słowy jeśli
# Random-car trafi na stażystę to Q-uber nie powinien
# przerywać swojej nauki. Naszym celem jest, żeby Q-uber
# dotarł do końca jak najkrótszą ścieżką.
# - sugerowałbym użyć dodatkowego kryterium stopu jak np.
# limit nagród czy maksymalna liczba kroków, zwłaszcza
# jeśli Państwa nagrody nie są ujemne tylko zerowe
