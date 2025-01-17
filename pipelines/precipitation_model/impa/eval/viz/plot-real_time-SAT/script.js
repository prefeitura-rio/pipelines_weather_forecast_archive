const images = document.querySelectorAll('.image-slider img');
const sliderBar = document.querySelector('.image-slider-bar');
const pauseButton = document.getElementById('pauseButton');
const resumeButton = document.getElementById('resumeButton');

let isCycling = true;
let intervalId;
let currentImage = 0;

function showImage(index) {
  images.forEach((image, i) => {
    image.classList.toggle('active', i === index);
  });
  sliderBar.value = index;
  currentImage = index;
  //isCycling = true; // Resume cycling after manual slider movement
}

function startCycling() {
  intervalId = setInterval(() => {
    currentImage = (currentImage + 1) % images.length;
    showImage(currentImage);
  }, 500); // Adjust interval as needed
}

function pauseCycling() {
  clearInterval(intervalId);
  isCycling = false;
}

function resumeCycling() {
  if (isCycling) return;
  startCycling();
  isCycling = true;
}

function handleSliderBarChange() {
  pauseCycling();
  showImage(Number(sliderBar.value));
}

startCycling(); // Start cycling initially

sliderBar.addEventListener('input', handleSliderBarChange);
pauseButton.addEventListener('click', pauseCycling);
resumeButton.addEventListener('click', resumeCycling);
