document.addEventListener('DOMContentLoaded', () => {
  //card options
  const cardArray = [
    {
      name: 'fries',
      img: 'fries.jpg'
    },
    {
      name: 'burger',
      img: 'burger.jpg'
    },
    {
      name: 'icecream',
      img: 'icecream.jpg'
    },
    {
      name: 'pizza',
      img: 'pizza.png'
    },
    {
      name: 'milkshake',
      img: 'milkshake.jpg'
    },
    {
      name: 'hotdog',
      img: 'hotdog.png'
    },
    {
      name: 'fries',
      img: 'fries.jpg'
    },
    {
      name: 'burger',
      img: 'burger.jpg'
    },
    {
      name: 'icecream',
      img: 'icecream.jpg'
    },
    {
      name: 'pizza',
      img: 'pizza.png'
    },
    {
      name: 'milkshake',
      img: 'milkshake.jpg'
    },
    {
      name: 'hotdog',
      img: 'hotdog.png'
    }
  ]

  cardArray.sort(() => 0.5 - Math.random())

  const grid = document.querySelector('.grid')
  const resultDisplay = document.querySelector('#result')
  var cardsChosen = []
  var cardsChosenId = []
  const cardsWon = []

  //create your board
  function createBoard() {
    for (let i = 0; i < cardArray.length; i++) {
      var card = document.createElement('img')
      card.setAttribute('src', 'blank.jpg')
      card.setAttribute('data-id', i)
      card.addEventListener('click', flipCard)
      grid.appendChild(card)
    }
  }

  //check for matches
  function checkForMatch() {
    var cards = document.querySelectorAll('img')
    const optionOneId = cardsChosenId[0]
    const optionTwoId = cardsChosenId[1]
    if (cardsChosen[0] === cardsChosen[1]) {
      alert('You found a match')
      cards[optionOneId].setAttribute('src', 'white.jpg')
      cards[optionTwoId].setAttribute('src', 'white.jpg')
      cardsWon.push(cardsChosen)
    } else {
      cards[optionOneId].setAttribute('src', 'blank.jpg')
      cards[optionTwoId].setAttribute('src', 'blank.jpg')
      alert('Sorry, try again')
    }
    cardsChosen = []
    cardsChosenId = []
    resultDisplay.textContent = cardsWon.length
    if  (cardsWon.length === cardArray.length/2) {
      resultDisplay.textContent = 'Congratulations! You found them all!'
    }
  }

  //flip your card
  function flipCard() {
    var cardId = this.getAttribute('data-id')
    cardsChosen.push(cardArray[cardId].name)
    cardsChosenId.push(cardId)
    this.setAttribute('src', cardArray[cardId].img)
    if (cardsChosen.length ===2) {
      setTimeout(checkForMatch, 500)
    }
  }

  createBoard()
})
