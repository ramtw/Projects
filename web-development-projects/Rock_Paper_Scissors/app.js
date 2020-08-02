let userScore=0;
let computerScore=0;
const userScore_span=document.getElementById("user-score");
const computerScore_span=document.getElementById("computer-score");
const scoreBoard_div=document.querySelector(".score-board");
const result_p=document.querySelector(".result > p");
const rock_div=document.getElementById("r");
const paper_div=document.getElementById("p");
const scissors_div=document.getElementById("s");


function getComputerChoice()
{
	const choice=['r','p','s'];
	const randomNumber=Math.floor(Math.random()*3);
	return choice[randomNumber];
}

function convertToWord(letter)
{
	if(letter=='r') return "rock";
	if(letter=='p') return "paper";
	if(letter=='s') return "scissors";
}
function win(userChoice,computerChoice)
{
	userScore++;
	userScore_span.innerHTML=userScore;
	computerScore_span.innerHTML=computerScore;
	result_p.innerHTML=convertToWord(userChoice)+" beats "+convertToWord(computerChoice)+". You win! 🔥️";
	document.getElementById(userChoice).classList.add("green-glow");
	setTimeout(function(){document.getElementById(userChoice).classList.remove("green-glow")},300);
}

function lose(userChoice,computerChoice)
{
	computerScore++;
	userScore_span.innerHTML=userScore;
	computerScore_span.innerHTML=computerScore;
	result_p.innerHTML=convertToWord(userChoice)+" loses to "+convertToWord(computerChoice)+". You lost....💩️";
	document.getElementById(userChoice).classList.add("red-glow");
	setTimeout(function(){document.getElementById(userChoice).classList.remove("red-glow")},300);
}

function draw(userChoice,computerChoice)
{
	userScore++;
	computerScore++;
	userScore_span.innerHTML=userScore;
	computerScore_span.innerHTML=computerScore;
	result_p.innerHTML=convertToWord(userChoice)+" draw "+convertToWord(computerChoice)+". Match draw! 🏳️";
	document.getElementById(userChoice).classList.add("gray-glow");
	setTimeout(function(){document.getElementById(userChoice).classList.remove("gray-glow")},300);
}

function game(userChoice)
{
	const compChoice=getComputerChoice();
	if(userChoice==compChoice)
	{
		draw(userChoice,compChoice);
	}
	else if((userChoice=='r' && compChoice=='s')||(userChoice=='p' && compChoice=='r')||(userChoice=='s' && compChoice=='p'))
	{
		win(userChoice,compChoice);
	}
	else
	{
		lose(userChoice,compChoice);
	}
}

function main()
{
	rock_div.addEventListener('click',function(){
		game("r");
	})
	paper_div.addEventListener('click',function(){
		game("p");
	})
	scissors_div.addEventListener('click',function(){
		game("s");
	})
}

main();

