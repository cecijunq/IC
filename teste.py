import os

film_names = []

list_direc = os.listdir("./data/plots_es/")

for movie in sorted(list_direc):
    movie = movie.split(".txt")[0]
    film_names.append(movie)

print(film_names)

# inglês
# ['12 Angry Men (1957 film)' '12 Years a Slave (film)' '127 Hours' '1917 (2019 film)' '42nd Street (film)' '49th Parallel (film)' '7th Heaven (1927 film)' 'A Beautiful Mind (film)' 'A Clockwork Orange (film)' 'A Farewell to Arms (1932 film)' 'A Few Good Men' 'A Letter to Three Wives' 'A Man for All Seasons (1966 film)' 'A Passage to India (film)' 'A Place in the Sun (1951 film)' 'A Room with a View (1985 film)' 'A Serious Man' "A Soldier's Story", 'A Star Is Born (1937 film)' 'A Star Is Born (2018 film)' 'A Streetcar Named Desire (1951 film)' 'Airport (1970 film)' 'Alfie (1966 film)' 'Alice Adams (1935 film)' 'All About Eve' 'All Quiet on the Western Front (1930 film)' 'All Quiet on the Western Front (2022 film)' 'All That Jazz (film)' 'All This, and Heaven Too' "All the King's Men (1949 film)", "All the President's Men (film)", 'Amadeus (film)' 'America America' 'American Beauty (1999 film)' 'American Graffiti' 'American Hustle' 'An American in Paris (film)' 'An Education' 'Anatomy of a Murder' 'Anchors Aweigh (film)' 'Annie Hall' 'Anthony Adverse' 'Apocalypse Now' 'Apollo 13 (film)' 'Argo (2012 film)' 'Around the World in 80 Days (1956 film)' 'Arrival (film)' 'Arrowsmith (film)' 'As Good as It Gets' 'Atlantic City (1980 film)' 'Atonement (2007 film)' 'Avatar (2009 film)' 'Avatar: The Way of Water' 'Awakenings' 'Babe (film)' 'Babel (film)' 'Barry Lyndon' 'Battleground (film)' 'Beasts of the Southern Wild' 'Beauty and the Beast (1991 film)' 'Ben-Hur (1959 film)' 'Birdman (film)' 'Black Panther (film)' 'Black Swan (film)' 'Bohemian Rhapsody (film)' 'Bonnie and Clyde (film)' 'Born Yesterday (1950 film)' 'Born on the Fourth of July (film)' 'Boyhood (2014 film)' 'Boys Town (film)' 'Braveheart' 'Bridge of Spies (film)' 'Broadcast News (film)' 'Broadway Melody of 1936' 'Brokeback Mountain' 'Bugsy' 'Butch Cassidy and the Sundance Kid' 'Cabaret (1972 film)' 'Call Me by Your Name (film)' 'Capote (film)' 'Captain Blood (1935 film)' 'Captain Phillips (film)' 'Captains Courageous (1937 film)' 'Casablanca (film)' 'Cat on a Hot Tin Roof (1958 film)' 'Cavalcade (1933 film)' 'Chariots of Fire' 'Chicago (2002 film)' 'Chinatown (1974 film)' 'Chocolat (2000 film)' 'Cimarron (1931 film)' 'Citizen Kane' 'Cleopatra (1963 film)' "Coal Miner's Daughter (film)", 'Coming Home (1978 film)' 'Crash (2004 film)' 'Cries and Whispers' 'Crossfire (film)' 'Crouching Tiger, Hidden Dragon' 'Dances with Wolves' 'Dangerous Liaisons' 'Dark Victory' 'Darkest Hour (film)' 'Darling (1965 film)' 'David Copperfield (1935 film)' 'Dead End (1937 film)' 'Decision Before Dawn' 'Deliverance' 'Disraeli (1929 film)' 'District 9' 'Django Unchained' 'Doctor Dolittle (1967 film)' 'Doctor Zhivago (film)' 'Dodsworth (film)' 'Dog Day Afternoon' "Don't Look Up", 'Double Indemnity' 'Dr. Strangelove' 'Driving Miss Daisy' 'Dunkirk (2017 film)' 'E.T. the Extra-Terrestrial' 'East Lynne (1931 film)' 'Elizabeth (film)' 'Elvis (2022 film)' 'Erin Brockovich (film)' 'Everything Everywhere All at Once' 'Extremely Loud & Incredibly Close (film)' 'Fanny (1961 film)' 'Fargo (1996 film)' 'Fatal Attraction' 'Father of the Bride (1950 film)' 'Fiddler on the Roof (film)' 'Field of Dreams' 'Finding Neverland (film)' 'Five Easy Pieces' 'Five Star Final' 'Flirtation Walk' 'For Whom the Bell Tolls (film)' 'Ford v Ferrari' 'Foreign Correspondent (film)' 'Forrest Gump' 'Four Weddings and a Funeral' 'Friendly Persuasion (1956 film)' 'From Here to Eternity' 'Frost_Nixon (film)' 'Funny Girl (film)' 'Gandhi (film)' 'Gangs of New York' 'Gaslight (1944 film)' "Gentleman's Agreement", 'Get Out' 'Ghost (1990 film)' 'Giant (1956 film)' 'Gigi (1958 film)' 'Gladiator (2000 film)' 'Going My Way' 'Gone with the Wind (film)' 'Good Night, and Good Luck' 'Good Will Hunting' 'Goodfellas' 'Gosford Park' 'Grand Hotel (1932 film)' 'Gravity (2013 film)' 'Green Book (film)' "Guess Who's Coming to Dinner", 'Hacksaw Ridge' 'Hamlet (1948 film)' 'Heaven Can Wait (1943 film)' 'Heaven Can Wait (1978 film)' 'Hello, Dolly! (film)' 'Henry V (1944 film)' 'Here Comes Mr. Jordan' 'Hold Back the Dawn' 'How Green Was My Valley (film)' 'How the West Was Won (film)' 'Howards End (film)' 'Hugo (film)' 'I Am a Fugitive from a Chain Gang' 'Imitation of Life (1934 film)' 'In Old Chicago' 'In Which We Serve' 'In the Bedroom' 'In the Heat of the Night (film)' 'In the Name of the Father (film)' 'Inception' 'Inglourious Basterds' 'It Happened One Night' "It's a Wonderful Life", 'Ivanhoe (1952 film)' 'JFK (film)' 'Jaws (film)' 'Jerry Maguire' 'Jezebel (1938 film)' 'Johnny Belinda (1948 film)' 'Joker (2019 film)' 'Judgment at Nuremberg' 'Julia (1977 film)' 'Juno (film)' "King Solomon's Mines (1950 film)", 'Kings Row' 'Kiss of the Spider Woman (film)' 'Kitty Foyle (film)' 'Kramer vs. Kramer' 'L.A. Confidential (film)' 'La Grande Illusion' 'La La Land' 'Lady for a Day' 'Lawrence of Arabia (film)' 'Lenny (film)' 'Les Misérables (2012 film)' 'Libeled Lady' 'Life Is Beautiful' 'Life of Pi (film)' 'Lilies of the Field (1963 film)' 'Lincoln (film)' 'Little Miss Sunshine' 'Little Women (1933 film)' 'Little Women (2019 film)' 'Lost Horizon (1937 film)' 'Lost in Translation (film)' 'Love Affair (1939 film)' 'Love Story (1970 film)' 'M*A*S*H (film)' 'Mad Max: Fury Road' 'Manchester by the Sea (film)' 'Mank' 'Marriage Story' 'Marty (film)' 'Mary Poppins (film)' 'Master and Commander: The Far Side of the World' 'Midnight Cowboy' 'Midnight Express (film)' 'Midnight in Paris' 'Mildred Pierce (film)' 'Milk (2008 American film)' 'Million Dollar Baby' 'Minari (film)' 'Miracle on 34th Street' 'Missing (1982 film)' 'Mississippi Burning' 'Mister Roberts (1955 film)' 'Moneyball (film)' 'Moonlight (2016 film)' 'Moonstruck' 'Moulin Rouge (1952 film)' 'Moulin Rouge!' 'Mr. Deeds Goes to Town' 'Mr. Smith Goes to Washington' 'Mrs. Miniver' 'Munich (2005 film)' 'Mutiny on the Bounty (1935 film)' 'Mutiny on the Bounty (1962 film)' 'My Fair Lady (film)' 'My Left Foot' 'Naughty Marietta (film)' 'Nebraska (film)' 'Network (1976 film)' 'Nicholas and Alexandra' 'Ninotchka' 'Nixon (film)' 'No Country for Old Men' 'Nomadland' 'Norma Rae' 'Of Mice and Men (1939 film)' 'Oliver! (film)' 'On Golden Pond (1981 film)' 'On the Waterfront' 'Once Upon a Time in Hollywood' "One Flew Over the Cuckoo's Nest (film)", 'One Hour with You' 'One Night of Love' 'Ordinary People' 'Out of Africa (film)' 'Patton (film)' 'Peyton Place (film)' 'Philomena (film)' 'Picnic (1955 film)' 'Places in the Heart' "Prizzi's Honor", 'Promising Young Woman' 'Pulp Fiction' 'Pygmalion (1938 film)' 'Quiz Show (film)' 'Quo Vadis (1951 film)' 'Raging Bull' 'Raiders of the Lost Ark' 'Rain Man' 'Ray (film)' 'Rebecca (1940 film)' 'Reds (film)' 'Roma (2018 film)' 'Roman Holiday' 'Saving Private Ryan' 'Sayonara' 'Scent of a Woman (1992 film)' "Schindler's List", 'Seabiscuit (film)' 'Selma (film)' 'Sense and Sensibility (film)' 'Separate Tables (film)' 'Sergeant York (film)' 'Seven Brides for Seven Brothers' 'Shakespeare in Love' 'Shane (film)' 'Shanghai Express (film)' 'She Done Him Wrong' 'Shine (film)' 'Silver Linings Playbook' 'Since You Went Away' 'Skippy (film)' 'Slumdog Millionaire' 'Sounder (film)' 'Spellbound (1945 film)' 'Spotlight (film)' 'Stage Door' 'Stagecoach (1939 film)' 'Star Wars (film)' 'Sunset Boulevard (film)' 'Suspicion (1941 film)' 'Taxi Driver' 'Tender Mercies' 'Terms of Endearment' 'Tess (1979 film)' 'Test Pilot (film)' 'The Accidental Tourist (film)' 'The Adventures of Robin Hood' 'The Alamo (1960 film)' 'The Apartment' 'The Artist (film)' 'The Aviator (2004 film)' 'The Awful Truth' 'The Banshees of Inisherin' 'The Barretts of Wimpole Street (1934 film)' 'The Best Years of Our Lives' 'The Big Chill (film)' 'The Big House (1930 film)' "The Bishop's Wife", 'The Blind Side (film)' 'The Bridge on the River Kwai' 'The Broadway Melody' 'The Caine Mutiny (film)' 'The Cider House Rules (film)' 'The Color Purple (1985 film)' 'The Conversation' 'The Crying Game' 'The Curious Case of Benjamin Button (film)' 'The Deer Hunter' 'The Defiant Ones' 'The Departed' 'The Descendants' 'The Diary of Anne Frank (1959 film)' 'The Divorcee' 'The Elephant Man (film)' 'The Emigrants (film)' 'The Exorcist' 'The Fabelmans' 'The Favourite' 'The Fighter' 'The French Connection (film)' 'The Front Page (1931 film)' 'The Fugitive (1993 film)' 'The Full Monty' 'The Godfather Part II' 'The Godfather Part III' 'The Godfather' 'The Good Earth (film)' 'The Graduate' 'The Grand Budapest Hotel' 'The Grapes of Wrath (film)' 'The Great Dictator' 'The Great Ziegfeld' 'The Greatest Show on Earth (film)' 'The Green Mile (film)' 'The Guns of Navarone (film)' 'The Heiress' 'The Help (film)' 'The Hours (film)' 'The Hurt Locker' 'The Imitation Game' 'The Insider (film)' 'The Irishman' 'The Kids Are All Right (film)' 'The Killing Fields (film)' "The King's Speech", 'The Last Emperor' 'The Letter (1940 film)' 'The Life of Emile Zola' 'The Little Foxes (film)' 'The Lives of a Bengal Lancer (film)' 'The Longest Day (film)' 'The Lord of the Rings: The Fellowship of the Ring' 'The Lord of the Rings: The Return of the King' 'The Lord of the Rings: The Two Towers' 'The Lost Weekend' 'The Magnificent Ambersons (film)' 'The Maltese Falcon (1941 film)' 'The Martian (film)' 'The Mission (1986 film)' 'The More the Merrier' 'The Music Man (1962 film)' "The Nun's Story (film)", 'The Ox-Bow Incident' 'The Patriot (1928 film)' 'The Philadelphia Story (film)' 'The Pianist (2002 film)' 'The Piano' 'The Pride of the Yankees' 'The Prince of Tides' 'The Private Life of Henry VIII' 'The Queen (2006 film)' 'The Quiet Man' 'The Racket (1928 film)' "The Razor's Edge (1946 film)", 'The Reader (2008 film)' 'The Red Shoes (1948 film)' 'The Remains of the Day (film)' 'The Revenant (2015 film)' 'The Robe (film)' 'The Rose Tattoo (film)' 'The Russians Are Coming, the Russians Are Coming' 'The Sand Pebbles (film)' 'The Shape of Water' 'The Shawshank Redemption' 'The Sixth Sense' 'The Smiling Lieutenant' 'The Snake Pit' 'The Social Network' 'The Song of Bernadette (film)' 'The Sound of Music (film)' 'The Sting' 'The Thin Red Line (1998 film)' 'The Towering Inferno' 'The Treasure of the Sierra Madre (film)' 'The Tree of Life (film)' 'The Turning Point (1977 film)' 'The Verdict' 'The Wizard of Oz' 'The Wolf of Wall Street (2013 film)' 'The Yearling (1946 film)' 'There Will Be Blood' 'Three Billboards Outside Ebbing, Missouri' 'Three Coins in the Fountain (film)' 'Titanic (1997 film)' 'To Kill a Mockingbird (film)' 'Tom Jones (1963 film)' 'Tootsie' 'Top Gun: Maverick' 'Top Hat' 'Toy Story 3' 'Trader Horn (1931 film)' 'Traffic (2000 film)' 'True Grit (2010 film)' 'Tár' 'Unforgiven' 'Up (2009 film)' 'Viva Villa!' 'War Horse (film)' 'Watch on the Rhine' 'West Side Story (1961 film)' 'West Side Story (2021 film)' 'Whiplash (2014 film)' 'Wings (1927 film)' "Winter's Bone", 'Witness (1985 film)' 'Witness for the Prosecution (1957 film)' 'Women Talking (film)' 'Working Girl' 'Wuthering Heights (1939 film)' 'Yankee Doodle Dandy' "You Can't Take It with You (film)", 'Z (1969 film)' 'Zorba the Greek (film)']