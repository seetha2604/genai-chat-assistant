let sessionId = localStorage.getItem("sessionId");

async function send(){

const input = document.getElementById("input");
const chat = document.getElementById("chat");

const message = input.value.trim();

if(message === "") return;

chat.innerHTML += "<p class='user'>You: " + message + "</p>";

const res = await fetch("/api/chat",{
method:"POST",
headers:{ "Content-Type":"application/json" },
body: JSON.stringify({
sessionId: sessionId,
message: message
})
});

const data = await res.json();

sessionId = data.sessionId;
localStorage.setItem("sessionId", sessionId);

chat.innerHTML += "<p class='bot'>Bot: " + data.reply + "</p>";

chat.scrollTop = chat.scrollHeight;

input.value="";
}

document.getElementById("input").addEventListener("keypress", function(event){
if(event.key === "Enter"){
event.preventDefault();
send();
}
});

function newChat(){
document.getElementById("chat").innerHTML = "";
sessionId = null;
localStorage.removeItem("sessionId");
}