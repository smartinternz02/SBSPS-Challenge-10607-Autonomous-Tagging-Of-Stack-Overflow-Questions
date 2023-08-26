const questionInput = document.getElementById('questionInput');
const submitButton = document.getElementById('submitButton');
const tagsContainer = document.getElementById('tagsContainer');
const tagsList = document.getElementById('tagsList');

submitButton.addEventListener('click', async () => {
  const question = questionInput.value;

  try {
    const response = await fetch(`https://api.stackexchange.com/2.3/tags?order=desc&min=1&max=5&sort=popular&site=stackoverflow`);
    const data = await response.json();

    if (data && data.items && data.items.length > 0) {
      const tags = data.items.map(item => item.name);
      displayTags(tags);
    } else {
      console.log('No tags found.');
    }
  } catch (error) {
    console.error('Error fetching data:', error);
  }
});

function displayTags(tags) {
  tagsList.innerHTML = '';
  tags.forEach(tag => {
    const tagItem = document.createElement('li');
    tagItem.textContent = tag;
    tagsList.appendChild(tagItem);
  });

  tagsContainer.classList.remove('hidden');
}
