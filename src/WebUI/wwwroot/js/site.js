const clamp = (num, min, max) => Math.min(Math.max(num, min), max)

const CellEvent = Object.freeze({
    CURSOR_IN: Symbol("CURSOR_IN"),
    CURSOR_OUT: Symbol("CURSOR_OUT"),
    DO_PAINT: Symbol("DO_PAINT")
});

const Grid = function (container) {
    this.Height = 0;
    this.Width = 0;
    this.Container = container;
    this.Cells = [];
    this.Display = [];
    this.OnResultsFetched = null;
    this.BrushSize = 4;
}

Grid.prototype = {
    Create(height, width, callback) {
        this.Height = height;
        this.Width = width;
        this.OnResultsFetched = callback;
        let index = 0;
        for (let y = 0; y < height; y++) {
            let row = [];
            for (let x = 0; x < width; x++) {

                var activeCell = new ActiveCell(this, index++, x, y);
                row.push(activeCell);
                this.Cells.push(activeCell);

            }
            this.Display.push(row);
        }
    },

    SubmitEvent() {
        var collectedInputs = this.Cells.map(cell => cell.Value);
        fetch('/api/infer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(collectedInputs)
        }).then(response => {
            response
                .json()
                .then(data => {
                    this.OnResultsFetched(data);
                });
        });
    }

}

const ActiveCell = function (grid, index, x, y) {
    this.Grid = grid;
    this.Index = index;
    this.X = x;
    this.Y = y;
    this.Value = 0;
    var element = document.createElement('div');
    element.draggable = false;
    element.innerHTML = index; // + '/' + this.Value;
    element.className = 'cell';
    grid.Container.appendChild(element);
    this.Cell = element;

    element.addEventListener('mouseenter', (event) => {

        this.ApplyBrush({ event: event, action: CellEvent.CURSOR_IN });
    });

    element.addEventListener('mouseleave', (event) => {

        this.ApplyBrush({ event: event, action: CellEvent.CURSOR_OUT });
    });

    element.addEventListener('mouseup', (event) => {
        this.ApplyBrush({ event: event, action: CellEvent.DO_PAINT });
        this.SubmitEvent();
        
    });
}

ActiveCell.prototype = {
    ApplyBrush(ev) {
        let minX = 0;
        let minY = 0;
        let maxX = this.Grid.Width - 1;
        let maxY = this.Grid.Height - 1;

        if (ev.event.buttons == 1) {
            ev.action = CellEvent.DO_PAINT;
        }

        let brushRadius = this.Grid.BrushSize;  

        for (let y = clamp(this.Y - brushRadius, minY, maxY); y <= clamp(this.Y + brushRadius, minY, maxY); y++) {
            for (let x = clamp(this.X - brushRadius, minX, maxX); x <= clamp(this.X + brushRadius, minX, maxX); x++) {
                const distance = Math.sqrt(Math.pow(this.Y - y, 2) + Math.pow(this.X - x, 2));
                if (distance <= brushRadius) {
                    let opacity = Math.pow(1 - (distance / brushRadius), 2);

                    x = Math.round(x);
                    y = Math.round(y);
                    var currentCell = this.Grid.Display[y][x];

                    if (opacity > 0.6)
                        opacity = 1;
                    if (opacity < 0.4)
                        opacity = 0;

                    switch (ev.action) {
                        case CellEvent.CURSOR_IN:
                            if (currentCell.Value != 0) break;
                            currentCell.Cell.style.backgroundColor = `rgba(255, 204, 0, ${opacity})`;
                            break;
                        case CellEvent.CURSOR_OUT:
                            if (currentCell.Value != 0) break;
                            currentCell.Cell.style.backgroundColor = '';
                            break;
                        case CellEvent.DO_PAINT:
                            let val = opacity;
                            if (currentCell.Value > val) break;
                            currentCell.Cell.style.backgroundColor = `rgba(0, 0, 0, ${opacity.toFixed(2)})`;
                            //currentCell.Cell.style.backgroundColor = `rgba(0, 0, 0, 100%})`;
                            currentCell.Value = val;
                            //currentCell.Cell.innerHTML = currentCell.Index + '/' + currentCell.Value.toFixed(2);
                            
                            break;
                        default:
                    }
                }
            }
        }
    },
    SubmitEvent() {
        this.Grid.SubmitEvent();
    }
}

const OutputGrid = function (container) {
    this.Container = container;
    this.Cells = [];
    var self = this;
}

OutputGrid.prototype = {
    Create(size) {
        for (let i = 0; i < size; i++) {
            var displayCell = new ResultDisplay(this.Container, i);
            this.Cells.push(displayCell);
        }
    },
    SetResults(results) {
        console.log(results);
        for (let i = 0; i < results.length; i++) {
            this.Cells[i].SetValue(results[i]*100);
        }
    }
}

const ResultDisplay = function (container, index) {
    this.Container = container;
    this.Index = index;

    const template = document.getElementById('result');

    const clone = template.content.cloneNode(true);

    this.ProgressBar = clone.querySelector('.progress-bar'); 
    this.Label = clone.querySelector('.badge');

    this.ProgressBar.setAttribute('aria-label', index);
    this.Label.innerText = index;

    this.Container.appendChild(clone);
}

ResultDisplay.prototype = {

    SetValue(value) {
        this.ProgressBar.setAttribute('style', 'width:' + value + '%');
    }

}

