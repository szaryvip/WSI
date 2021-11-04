"""
Witam Państwa, 

Tematem trzecich ćwiczeń są dwuosobowe gry deterministyczne. Państwa zadaniem będzie napisanie programu / skryptu, który buduje drzewo zadanej gry a następnie gra sam ze sobą wykorzystując do tego algorytm minimax. Należałoby sprawdzić dwa przypadki: 

     jeden z graczy gra w sposób losowy (tzn. nie używa naszego algorytmu) a drugi stara się optymalizować swoje ruchy (random vs minimax) 
    Obaj gracze podejmują optymalne decyzje (minimax vs minimax) 


Aktualny stan planszy powinien być wyświetlany w konsoli, ale mogą Państwo użyć dodatkowych bibliotek do pisania i wyświetlania gier w Pythonie takich jak pygame.
W raporcie należałoby przedstawić przykład wykonania programu wraz z odpowiadającymi stanami drzewa gry i wyborami algorytmu minimax. Zastosowanie przycinania alfa-beta nie jest częścią zadania, ale może okazać się potrzebne zależnie od otrzymanego typu gry. Proszę wykazać w raporcie wygraną każdej ze stron, czyli że gra nie jest ustawiona. 
Program buduje drzewo gry w Czwórki. Wejściem są wymiary planszy NxM (N>= 5, M >=4) oraz maksymalna głębokość drzewa.
Zasady:
- gracze co turę wrzucają 1 kolorowy token (kolor odpowiada kolorowi danego gracza) do jednej z kolumn na planszy
- jeżeli w pionie, poziomie lub na ukos znajdą się 4 tokeny gracza to ten gracz wygrywa
- jeżeli plansza się zapełni to dochodzi do remisu 

- kiedy budujemy drzewo minimax o głębokości D to warto sprawdzić różne wartości tego D<=D_Max
- należałoby sprawdzić, czy gra nie faworyzuje, któregoś z graczy i wykazać w raporcie wygraną obu stron. Jeżeli istnieje możliwość remisu to też warto byłoby to pokazać.
- gry odbywają się w sposób losowy, dlatego też na potrzebę pomiarów należałoby wykonać N prób

- wybór następnego ruchu przez algorytm dla zbioru tak samo ocenionych stanów powinien wykonywać się w sposób losowy (tzn. jeżeli mamy do wyboru pole 1,2 i 5 o tej samej wartości funkcji heurystycznej to każde z nich powinno mieć szansę wyboru a nie tylko pole nr 1) """